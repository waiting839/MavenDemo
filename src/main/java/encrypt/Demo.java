package encrypt;

/**
 * @author 吴嘉烺
 * @description
 * @date 2022/11/28
 */
public class Demo {
    public static void main(String[] args) {
        DefaultEncryptUtil defaultEncryptUtil = new DefaultEncryptUtil();
        String s = defaultEncryptUtil.decrypt("19a2a2f87ddd128632d580ad2cb015ac");
        System.out.println(s);
    }
}
