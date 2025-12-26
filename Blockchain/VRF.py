import hashlib
import hmac


class ECVRF:
    """ 可验证随机函数实现：保障记账节点选取的随机性与公平性 """

    @staticmethod
    def prove(private_key, alpha: bytes):
        """ 生成证明 pi 和 随机输出 beta """
        pi = private_key.sign(alpha)  # 基于私钥的确定性签名
        beta = hashlib.sha512(pi + b"AGRO_VRF_2025").digest()  # 映射为随机值
        return beta, pi

    @staticmethod
    def verify(pub_key_bytes: bytes, alpha: bytes, pi: bytes, beta: bytes):
        """ 验证 VRF 证明的有效性 """
        from cryptography.hazmat.primitives.asymmetric import ed25519
        try:
            pk = ed25519.Ed25519PublicKey.from_public_bytes(pub_key_bytes)
            pk.verify(pi, alpha)
            expected_beta = hashlib.sha512(pi + b"AGRO_VRF_2025").digest()
            return hmac.compare_digest(expected_beta, beta)
        except:
            return False
