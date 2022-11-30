package encrypt;


/**
 * @author lfq
 * @description 加解密
 * @date 2022/6/16
 */
public class CryptoUtil {

    private static final byte[] SECRET_KEY = {'a','b','c','d','e','f','1','2','3','4','5','6','7','8','9','0'};
    /**
     * 加密 每次加密都一样
     * @param value 需要加密的值
     * @return
     */
    public static String encrypt(String value)
    {
        byte[] needEn = ByteUtil.str2Byte(value);
        byte[] alreadyEn = EncryptUtil.AesEncrypt(needEn,SECRET_KEY);
        return ByteUtil.bytes2Hex(alreadyEn);
    }

    /**
     * 解密
     * @param value 需要解密的值
     * @return
     */
    public static String decrypt(String value)
    {
        byte[] de = ByteUtil.hex2Bytes(value);
        byte[] alreadyDe = EncryptUtil.AesDecrypt(de,SECRET_KEY);
        return ByteUtil.byte2Str(alreadyDe);
    }

    /**
     * md5加密加盐
     * @param salt 盐值
     * @param sha1 已经经过sha1加密的密码（全大写）
     * @return md5加密加盐之后的密码（转小写）
     */
    public static String encodeMd5Password(String salt, String sha1)
    {
        String md5 = ByteUtil.bytes2Hex(EncryptUtil.MD5(sha1.toUpperCase())).toLowerCase();
        return ByteUtil.bytes2Hex(EncryptUtil.MD5(md5 + salt)).toLowerCase();
    }

    /**
     * md5加盐
     * @param salt 盐值
     * @param md5 已经md5加密过的密码，全小写
     * @return md5加密加盐之后的密码（转小写）
     */
    public static String encodeSaltPassword(String salt, String md5)
    {
        return ByteUtil.bytes2Hex(EncryptUtil.MD5(md5 + salt)).toLowerCase();
    }
}
