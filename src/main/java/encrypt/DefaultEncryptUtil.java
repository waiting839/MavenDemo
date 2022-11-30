package encrypt;

import org.apache.commons.lang3.StringUtils;

/**
 * 默认加解密接口
 * @author lfq
 * @date 2022/8/29
 */
public class DefaultEncryptUtil{

    /** 不可逆加密。为了兼容以前旧的密码没有加盐，加盐的密码加前缀，在密码校验那里对没有加盐的密码不加盐处理
     * 如果是original值，先sha1，再md5；
     * 如果是sha1，直接md5；
     * 如果是md5，原样返回 */
//    public String hash(String sha1, String salt, HashEncodeType type) {
//        if(type==HashEncodeType.md5) {
//            return sha1.toLowerCase();
//        }
//        if(type==HashEncodeType.original) {
//            sha1 = ConvertUtil.byteToString(EncryptUtil.Sha1(sha1));
//        }
//        if(!CheckUtil.isNullorEmpty(salt)) {
//            sha1 = sha1 + EncryptInterface.SIGN + salt;
//            return ENC_PREFIX + ConvertUtil.byteToString(EncryptUtil.MD5(sha1.toUpperCase())).toLowerCase() + ENC_SUFFIX;
//        }
//        return ConvertUtil.byteToString(EncryptUtil.MD5(sha1.toUpperCase())).toLowerCase();
//    }

    /** 可逆加密 */
    public String encrypt(String value) {
        return "enc(" + CryptoUtil.encrypt(value) + ")";
    }

    /** 解密 */
    public String decrypt(String value) {
        if (isEncrypted(value)) {
            value = value.substring(4, value.length() - 1);
        }
        return CryptoUtil.decrypt(value);
    }

    /**
     * 是否已经加密过
     * @param value
     * @return
     */
    public boolean isEncrypted(String value) {
        if (StringUtils.isNotBlank(value) && value.startsWith("enc(") && value.endsWith(")")){
            return true;
        }
        return false;
    }


}
