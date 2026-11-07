"""CONV row bundles / FC three-row column bundles; bounded-bank retry timing."""

from .reference_psum import SharedPsumPool as ReferencePsumPool


class SharedPsumPool(ReferencePsumPool):
    def _backoff(self):
        # 缓冲退休路径由 bank 轮询保证进展：被拒后下一拍即可重试，不做多拍退避。
        pass

