
from concurrent.futures import ThreadPoolExecutor


class EvalThreadPools:
    def __init__(
        self,
        n_jobs_pool_llm: int,
        n_jobs_pool_csim: int,
        n_jobs_pool_synth: int,
        n_jobs_pool_hlsfactory: int,
        tokens_per_minute: int | None = None,
        requests_per_minute: int | None = None,
    ) -> None:
        self.n_jobs_pool_llm = n_jobs_pool_llm
        self.n_jobs_pool_csim = n_jobs_pool_csim
        self.n_jobs_pool_synth = n_jobs_pool_synth
        self.n_jobs_pool_hlsfactory = n_jobs_pool_hlsfactory

        self.tokens_per_minute = tokens_per_minute
        self.requests_per_minute = requests_per_minute

        if n_jobs_pool_llm <= 1:
            raise ValueError("n_jobs_pool_llm must be greater than 1")
        if n_jobs_pool_csim <= 1:
            raise ValueError("n_jobs_pool_csim must be greater than 1")
        if n_jobs_pool_synth <= 1:
            raise ValueError("n_jobs_pool_synth must be greater than 1")
        if n_jobs_pool_hlsfactory <= 1:
            raise ValueError("n_jobs_pool_hlsfactory must be greater than 1")
            
        self.pool_llm = ThreadPoolExecutor(max_workers=n_jobs_pool_llm)
        self.pool_csim = ThreadPoolExecutor(max_workers=n_jobs_pool_csim)
        self.pool_synth = ThreadPoolExecutor(max_workers=n_jobs_pool_synth)
        self.pool_hlsfactory = ThreadPoolExecutor(max_workers=n_jobs_pool_hlsfactory)

    def shutdown(self):
        self.pool_llm.shutdown(wait=True)
        self.pool_csim.shutdown(wait=True)
        self.pool_synth.shutdown(wait=True)
        self.pool_hlsfactory.shutdown(wait=True)