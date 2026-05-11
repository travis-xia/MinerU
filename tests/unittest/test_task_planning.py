from pathlib import Path

from mineru.cli.client import InputDocument, plan_tasks


def _build_doc(name: str, pages: int, order: int, ocr_classify: bool | None = None) -> InputDocument:
    return InputDocument(
        path=Path(f"/tmp/{name}.pdf"),
        suffix="pdf",
        stem=name,
        effective_pages=pages,
        order=order,
        ocr_classify=ocr_classify,
    )


def test_hybrid_plan_tasks_separates_ocr_groups():
    documents = [
        _build_doc("txt_a", 6, 0, False),
        _build_doc("ocr_a", 4, 1, True),
        _build_doc("txt_b", 3, 2, False),
        _build_doc("ocr_b", 2, 3, True),
    ]

    planned_tasks = plan_tasks(
        documents=documents,
        backend="hybrid-auto-engine",
        processing_window_size=10,
        parse_method="auto",
    )

    assert len(planned_tasks) == 2
    assert [[doc.stem for doc in task.documents] for task in planned_tasks] == [
        ["txt_a", "txt_b"],
        ["ocr_a", "ocr_b"],
    ]
    assert all(
        len({doc.ocr_classify for doc in task.documents}) == 1
        for task in planned_tasks
    )


def test_hybrid_plan_tasks_allows_double_window_for_oversized_doc():
    documents = [
        _build_doc("large", 11, 0, False),
        _build_doc("small_a", 5, 1, False),
        _build_doc("small_b", 4, 2, False),
    ]

    planned_tasks = plan_tasks(
        documents=documents,
        backend="hybrid-auto-engine",
        processing_window_size=10,
        parse_method="auto",
    )

    assert len(planned_tasks) == 1
    assert planned_tasks[0].total_pages == 20
    assert [doc.stem for doc in planned_tasks[0].documents] == [
        "large",
        "small_a",
        "small_b",
    ]


def test_pipeline_plan_tasks_keeps_single_large_doc_isolated():
    documents = [
        _build_doc("large", 11, 0),
        _build_doc("small", 5, 1),
    ]

    planned_tasks = plan_tasks(
        documents=documents,
        backend="pipeline",
        processing_window_size=10,
        parse_method="auto",
    )

    assert [task.total_pages for task in planned_tasks] == [11, 5]
    assert [[doc.stem for doc in task.documents] for task in planned_tasks] == [
        ["large"],
        ["small"],
    ]
