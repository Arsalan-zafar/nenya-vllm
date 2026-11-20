"""Shared by the scripts that run vLLM offline."""

import atexit


def shutdown_llm(llm) -> None:
    """Stop the engine explicitly before the interpreter tears down. Leaving it to interpreter teardown occasionally
    aborts the process ("terminate called without an active exception", exit code 134) after all work is done."""
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:  # noqa: BLE001 - best effort at exit
        print(f"engine shutdown: {e!r}")


def make_llm(**kwargs):
    """vllm.LLM(**kwargs), shut down by an atexit hook (runs while the interpreter is still whole)."""
    from vllm import LLM

    llm = LLM(**kwargs)
    atexit.register(shutdown_llm, llm)
    return llm
