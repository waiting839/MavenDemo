package encrypt;

import org.apache.commons.lang3.StringUtils;

import java.io.UnsupportedEncodingException;

/**
 * @author lfq
 * @description byte与字符串互转
 * @date 2022/6/16
 */
public class ByteUtil {

    private static final char[] DIGITS_HEX = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd',
            'e', 'f' };
    private static final String HEX_DIGITS = "0123456789abcdef";

    /**
     * 任意字符转byte
     * @param str
     * @return
     */
    public static byte[] str2Byte(String str){
        if (StringUtils.isNotBlank(str)) {
            try {
                String hex = toHex(str.getBytes("utf-8"));
                return hex2Bytes(hex);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return null;
    }

    /**
     * byte转任意字符
     * @param bytes
     * @return
     */
    public static String byte2Str(byte[] bytes){
        if (bytes != null) {
            try {
                String hex = bytes2Hex(bytes);
                return fromHex(hex.toCharArray());
            } catch (UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
        return null;
    }
    /**
     * byte转为16进制字符
     * @param src
     * @return
     */
    public static String bytes2Hex(byte[] src) {
        if (src == null || src.length <= 0) {
            return null;
        }
        char[] res = new char[src.length * 2]; // 每个byte对应两个字符
        for (int i = 0, j = 0; i < src.length; i++) {
            res[j++] = DIGITS_HEX[src[i] >> 4 & 0x0f]; // 先存byte的高4位
            res[j++] = DIGITS_HEX[src[i] & 0x0f]; // 再存byte的低4位
        }
        return new String(res);
    }

    /**
     * 16进制字符转为byte
     * @param hexString
     * @return
     */
    public static byte[] hex2Bytes(String hexString) {
        if (hexString == null || hexString.equals("")) {
            return null;
        }
        int length = hexString.length() / 2;
        char[] hexChars = hexString.toCharArray();
        byte[] bytes = new byte[length];
        for (int i = 0; i < length; i++) {
            int pos = i * 2; // 两个字符对应一个byte
            int h = HEX_DIGITS.indexOf(hexChars[pos]) << 4; // 注1
            int l = HEX_DIGITS.indexOf(hexChars[pos + 1]); // 注2
            if(h == -1 || l == -1) { // 非16进制字符
                return null;
            }
            bytes[i] = (byte) (h | l);
        }
        return bytes;
    }

    /**
     * 非16进制字符转16进制字符
     * @param data
     * @return
     */
    public static String toHex(byte[] data) {
        int l = data.length;
        char[] out = new char[l << 1];
        for (int i = 0, j = 0; i < l; i++) {
            out[j++] = DIGITS_HEX[(0xF0 & data[i]) >>> 4];
            out[j++] = DIGITS_HEX[0x0F & data[i]];
        }
        return new String(out);
    }

    /**
     * 16进制字符转非16进制字符
     * @param data
     * @return
     */
    public static String fromHex(char[] data) throws UnsupportedEncodingException {
        int len = data.length;
        if ((len & 0x01) != 0) {
            throw new RuntimeException("字符个数应该为偶数");
        }
        byte[] out = new byte[len >> 1];
        for (int i = 0, j = 0; j < len; i++) {
            int f = toDigit(data[j], j) << 4;
            j++;
            f |= toDigit(data[j], j);
            j++;
            out[i] = (byte) (f & 0xFF);
        }
        return new String(out,"utf-8");
    }

    protected static int toDigit(char ch, int index) {
        int digit = Character.digit(ch, 16);
        if (digit == -1) {
            throw new RuntimeException("Illegal hexadecimal character " + ch + " at index " + index);
        }
        return digit;
    }

}
