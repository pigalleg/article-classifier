from scripts.util.execution_time_logger import AffinityTimingLogger


def test_timing_logger_writes_token_columns(tmp_path):
    logger = AffinityTimingLogger.from_results_dir(tmp_path)

    logger.log_event(
        stage="ra",
        abstract_index="ALL",
        document_title="ALL_ABSTRACTS",
        targets_evaluated=10,
        records_written=10,
        elapsed_seconds=2.5,
        input_tokens=120,
        output_tokens=80,
        total_tokens=200,
    )

    content = (tmp_path / "affinity_timing_log.csv").read_text(encoding="utf-8").splitlines()
    assert content
    assert content[0].endswith(",input_tokens,output_tokens,total_tokens")
    assert ",120,80,200" in content[1]
