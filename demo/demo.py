# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import os
import tempfile
from pathlib import Path

import click
import httpx

from mineru.cli import api_client as _api_client
from mineru.cli.api_protocol import (
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_PROCESSING_WINDOW_SIZE,
)
from mineru.cli.client import (
    HybridDependencyError,
    LocalAPIServer,
    PlannedTask,
    TaskFailure,
    collect_input_documents,
    download_result_zip,
    ensure_backend_dependencies,
    execute_planned_tasks,
    fetch_server_health,
    format_task_label,
    plan_tasks,
    resolve_effective_max_concurrent_requests,
    resolve_submit_concurrency,
    safe_extract_zip,
    submit_task,
    wait_for_local_api_ready,
)
from mineru.utils.config_reader import (
    get_max_concurrent_requests as read_max_concurrent_requests,
)


def build_form_data(
    language: str,
    backend: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    server_url: str | None,
    start_page_id: int,
    end_page_id: int | None,
    *,
    return_layout_pdf: bool = False,
    return_span_pdf: bool = False,
) -> dict[str, str | list[str]]:
    return _api_client.build_parse_request_form_data(
        lang_list=[language],
        backend=backend,
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
        server_url=server_url,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        return_md=True,
        return_middle_json=False,
        return_model_output=False,
        # 为 True 时才会生成 *_content_list.json / *_content_list_v2.json 并打入 ZIP。
        return_content_list=True,
        return_images=True,
        response_format_zip=True,
        return_original_file=True,
        return_layout_pdf=return_layout_pdf,
        return_span_pdf=return_span_pdf,
    )


def format_status_message(status_snapshot: _api_client.TaskStatusSnapshot) -> str:
    if status_snapshot.queued_ahead is None:
        return status_snapshot.status
    return f"{status_snapshot.status} (queued_ahead={status_snapshot.queued_ahead})"


def prepare_local_api_temp_dir() -> None:
    current_temp_dir = Path(tempfile.gettempdir())
    if os.name == "nt" or not Path("/tmp").exists():
        return
    if not str(current_temp_dir).startswith("/mnt/"):
        return

    # vLLM/ZeroMQ IPC sockets fail on drvfs-backed temp directories under WSL.
    os.environ["TMPDIR"] = "/tmp"
    tempfile.tempdir = None


def _format_failures(failures: list[TaskFailure]) -> str:
    return "\n".join(
        f"- task#{f.task_index} ({', '.join(f.document_stems)}): {f.message}"
        for f in sorted(failures, key=lambda item: item.task_index)
    )


async def run_planned_task_parallel(
    http_client: httpx.AsyncClient,
    server_health: _api_client.ServerHealth,
    planned_task: PlannedTask,
    form_data: dict[str, str | list[str]],
    output_path: Path,
) -> None:
    """单次解析任务：与 mineru CLI 的 run_planned_task 一致（无可视化、无 live 状态条）。"""
    submit_response = await submit_task(
        client=http_client,
        base_url=server_health.base_url,
        planned_task=planned_task,
        form_data=form_data,
    )
    label = format_task_label(planned_task)
    print(f"submitted {label} -> task_id={submit_response.task_id}")

    last_status_message: str | None = None

    def on_status_update(status_snapshot: _api_client.TaskStatusSnapshot) -> None:
        nonlocal last_status_message
        message = format_status_message(status_snapshot)
        if message == last_status_message:
            return
        last_status_message = message
        print(f"{label} status: {message}")

    await _api_client.wait_for_task_result(
        client=http_client,
        submit_response=submit_response,
        task_label=label,
        status_snapshot_callback=on_status_update,
    )
    print(f"{label} status: completed")

    zip_path = await download_result_zip(
        client=http_client,
        submit_response=submit_response,
        planned_task=planned_task,
    )
    try:
        safe_extract_zip(zip_path, output_path)
    finally:
        zip_path.unlink(missing_ok=True)


async def run_demo(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    api_url: str | None = None,
    backend: str = "hybrid-auto-engine",
    parse_method: str = "auto",
    language: str = "ch",
    formula_enable: bool = True,
    table_enable: bool = True,
    server_url: str | None = None,
    start_page_id: int = 0,
    end_page_id: int | None = None,
    limit: int | None = None,
    return_layout_pdf: bool = False,
    return_span_pdf: bool = False,
) -> None:
    api_url = api_url or None
    server_url = server_url or None
    if backend.endswith("http-client") and not server_url:
        raise ValueError(f"backend={backend} requires server_url")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer or None")

    input_path_resolved = Path(input_path).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if api_url is None:
        try:
            ensure_backend_dependencies(backend)
        except HybridDependencyError as exc:
            raise ValueError(str(exc)) from exc

    try:
        documents = collect_input_documents(
            input_path_resolved,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
    except click.ClickException as exc:
        raise ValueError(str(exc)) from exc

    if limit is not None:
        total = len(documents)
        documents = documents[:limit]
        if not documents:
            raise ValueError(f"limit={limit} leaves no files to process (had {total})")
        if len(documents) < total:
            print(
                f"limit={limit}: processing {len(documents)} of {total} document(s) "
                "(see collect_input_documents order)"
            )

    form_data = build_form_data(
        language=language,
        backend=backend,
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
        server_url=server_url,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        return_layout_pdf=return_layout_pdf,
        return_span_pdf=return_span_pdf,
    )

    local_server: LocalAPIServer | None = None

    async with httpx.AsyncClient(
        timeout=_api_client.build_http_timeout(),
        follow_redirects=True,
    ) as http_client:
        try:
            if api_url is None:
                prepare_local_api_temp_dir()
                local_server = LocalAPIServer()
                base_url = local_server.start()
                print(f"Started local mineru-api: {base_url}")
                server_health = await wait_for_local_api_ready(http_client, local_server)
                effective_max_concurrent_requests = server_health.max_concurrent_requests
            else:
                server_health = await fetch_server_health(
                    http_client,
                    _api_client.normalize_base_url(api_url),
                )
                effective_max_concurrent_requests = resolve_effective_max_concurrent_requests(
                    read_max_concurrent_requests(default=DEFAULT_MAX_CONCURRENT_REQUESTS),
                    server_health.max_concurrent_requests,
                )

            # 与 mineru CLI run_orchestrated_cli 一致：拿到 /health 后再 plan（pipeline / hybrid 依赖 processing_window_size）。
            processing_window_size = (
                server_health.processing_window_size
                if backend == "pipeline" or backend.startswith("hybrid-")
                else DEFAULT_PROCESSING_WINDOW_SIZE
            )
            planned_tasks = plan_tasks(
                documents=documents,
                backend=backend,
                processing_window_size=processing_window_size,
                parse_method=parse_method,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
            )

            print(f"Using API: {server_health.base_url}")
            concurrency = resolve_submit_concurrency(
                effective_max_concurrent_requests,
                len(planned_tasks),
            )
            print(
                f"Parallel submit: {concurrency} worker(s), {len(planned_tasks)} task(s), "
                f"effective_max_concurrent_requests={effective_max_concurrent_requests}"
            )

            failures = await execute_planned_tasks(
                planned_tasks=planned_tasks,
                concurrency=concurrency,
                task_runner=lambda pt: run_planned_task_parallel(
                    http_client,
                    server_health,
                    pt,
                    form_data,
                    output_path,
                ),
            )
            if failures:
                raise RuntimeError(
                    f"{len(failures)} task(s) failed:\n{_format_failures(failures)}"
                )
        finally:
            if local_server is not None:
                local_server.stop()

    print(f"Extracted result(s) to: {output_path}")


def main() -> None:
    # 本地 models-dir 依赖环境变量 MINERU_MODEL_SOURCE=local；mineru.json 里的 "model-source" 不会被读取。
    # setdefault 不覆盖已在 shell 中 export 的值。
    os.environ.setdefault("MINERU_MODEL_SOURCE", "local")
    # API 客户端并发请求上限（与 mineru.json 无关，需用环境变量）。
    os.environ.setdefault("MINERU_API_MAX_CONCURRENT_REQUESTS", "4")
    # hybrid / VLM / pipeline 每批处理的最大页数（get_processing_window_size）；不覆盖 shell 里已 export 的值。
    os.environ.setdefault("MINERU_PROCESSING_WINDOW_SIZE", "80")
    # vLLM 引擎 gpu_memory_utilization（0~1）；仅在未通过 mineru-api 额外参数显式传入时生效。
    os.environ.setdefault("MINERU_VLLM_GPU_MEMORY_UTILIZATION", "0.7")
    # 指定设备用环境变量 MINERU_DEVICE_MODE（如 cuda）；与 json 里的 device-mode 无关，需要时可取消注释：
    # os.environ.setdefault("MINERU_DEVICE_MODE", "cuda")

    # 使用绝对路径
    input_path = Path("/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/lava/test_pdfs/test_pdfs")
    output_dir = Path("output/output_test_demo")

    # 若已手动启动 mineru-api，可设为 "http://127.0.0.1:8000"；None 则自动起临时本地服务。
    api_url = None

    # "hybrid-auto-engine" | "pipeline" | "vlm-auto-engine" | "*-http-client" 等
    backend = "hybrid-auto-engine"
    # "auto" | "txt" | "ocr"
    parse_method = "auto"
    # 与 CLI --lang 一致；pipeline / hybrid 的 OCR 语言提示
    language = "japan"
    # Enable formula parsing in the output.
    formula_enable = True
    # Enable table parsing in the output.
    table_enable = True
    # Required only for "*-http-client" backends, for example:
    # "http://127.0.0.1:30000"
    server_url = None
    # Zero-based page range. Set end_page_id to None to parse to the last page.
    start_page_id = 0
    end_page_id = None
    # 目录输入时最多处理前 N 个文件（按文件名排序后截取）；单文件或 None 表示不限制。
    # limit: int | None = None
    limit = 40

    # 需要 mineru-api 生成并打包 *_layout.pdf / *_span.pdf 时改为 True（默认 False，省时间与体积）。
    return_layout_pdf = False
    return_span_pdf = False

    # 若要从远端拉模型而非使用本地 models-dir，请在运行前 export MINERU_MODEL_SOURCE=huggingface
    # 或 modelscope，并注释掉上面的 setdefault("local", ...)。

    asyncio.run(
        run_demo(
            input_path=input_path,
            output_dir=output_dir,
            api_url=api_url,
            backend=backend,
            parse_method=parse_method,
            language=language,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            limit=limit,
            return_layout_pdf=return_layout_pdf,
            return_span_pdf=return_span_pdf,
        )
    )


if __name__ == "__main__":
    main()
