package encrypt;


import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

//import com.southgis.ibase.utils.exception.ServiceException;

/**
 * 加解密工具类：摘要加密、对称加解密、非对称加解密
 * 
 * @author HuangLeibing
 *
 */
public final class EncryptUtil
{

    private static final char[] HEX_DIGITS = {'0', '1', '2', '3', '4', '5',  
            '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};  

	/**
	 * MD5加密
	 * 
	 * @param input 要加密的字符串
	 * @return 如果成功，返回加密串；否则返回null
	 */
	public final static byte[] MD5(String input)
	{
		try
		{
			MessageDigest mdInst = MessageDigest.getInstance("MD5");
			return mdInst.digest(input.getBytes(StandardCharsets.UTF_8));
		}
		catch(NoSuchAlgorithmException e)
		{
			return null;
		}
	}
	
	/**
	 * sha1加密
	 * @param input 要加密的字符串
	 * @return 如果成功，返回加密串；否则返回null
	 */
	public final static byte[] Sha1(String input)
	{
		try
		{
			MessageDigest mDigest = MessageDigest.getInstance("SHA1");
			return mDigest.digest(input.getBytes(StandardCharsets.UTF_8));
		}
		catch(NoSuchAlgorithmException e)
		{
			return null;
		}
	}

	/**
	 * sha1加密 
	 * @param input 要加密的字符串
	 * @return 如果成功，返回小写加密字符串；否则返回null
	 */
	public static String encodeSha1(String input) {
		if (input == null) {
			return null;
		}
		try {
			MessageDigest messageDigest = MessageDigest.getInstance("SHA1");
			messageDigest.update(input.getBytes());
			return getFormattedText(messageDigest.digest());
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Takes the raw bytes from the digest and formats them correct.
	 * 
	 * @param bytes
	 *            the raw bytes from the digest.
	 * @return the formatted bytes.
	 */
	private static String getFormattedText(byte[] bytes) {
		int len = bytes.length;
		StringBuilder buf = new StringBuilder(len * 2);
		// 把密文转换成十六进制的字符串形式
		for (int j = 0; j < len; j++) {
			buf.append(HEX_DIGITS[(bytes[j] >> 4) & 0x0f]);
			buf.append(HEX_DIGITS[bytes[j] & 0x0f]);
		}
		return buf.toString();
	}
	
	/**
	 * 对称加密数据
	 * 
	 * @param input
	 * @param sPwd 密码最大长度32个字节（一个英文一个字节，一个中文三个字节），超过32的密钥值将会舍去。
	 * @return
	 */
	public static byte[] AesEncrypt(byte[] input, String sPwd)
	{
		return AesCipher(input,sPwd.getBytes(StandardCharsets.UTF_8),true);
	}
	
	/**
	 * 对称加密数据
	 * @param input
	 * @param sPwd 密码最大长度32个字节，超过32的密钥值将会舍去。
	 * @return
	 */
	public static byte[] AesEncrypt(byte[] input, byte[] sPwd)
	{
		return AesCipher(input,sPwd,true);
	}
	/**
	 * 对称解密数据
	 * @param input
	 * @param sPwd 密码最大长度32个字节（一个英文一个字节，一个中文三个字节），超过32的密钥值将会舍去。
	 * @return
	 */
	public static byte[] AesDecrypt(byte[] input, String sPwd)
	{
		return AesCipher(input,sPwd.getBytes(StandardCharsets.UTF_8),false);
	}
	/**
	 * 对称解密数据
	 * @param input
	 * @param sPwd 密码最大长度32个字节，超过32的密钥值将会舍去。
	 * @return
	 */
	public static byte[] AesDecrypt(byte[] input, byte[] sPwd)
	{
		return AesCipher(input,sPwd,false);
	}
	
	/**
	 * 将验证用户信息加密，做为请求头X-AUTH-ID的内容传递给内部调用的服务
	 * @param userId 当前用户ID
	 * @param rightData 权限信息（格式：权限值-操作列表），操作列表项以::分隔。如果传入空，则默认为127-*
	 * @param key 加密密钥，通过配置项“securityFilter.aes-key”获取
	 * @return 加密后的串，可设置给请求头X-AUTH-ID的内容
	 */
//	public static String AesUserIdHeader(String userId, String rightData, String key)
//	{
//		Map<String, Object> idInfo=new HashMap<>();
//		//添加干扰数据，以防相同内容加密得到相同结果
//		Random rand=new Random();
//		idInfo.put("id", rand.nextInt()+"``"+userId);
//		idInfo.put("security", rand.nextInt()+"``"+rightData);
//		idInfo.put("time", (new Date()).getTime());
//		String src=JsonUtil.toJsonString(idInfo);
//		byte[] byResult = EncryptUtil.AesEncrypt(src.getBytes(StandardCharsets.UTF_8), key);
//		return "**"+Base64.getUrlEncoder().encodeToString(byResult);
//	}
	
	private static byte[] AesCipher(byte[] input,byte[] password,boolean encrypt)
	{
		try{
			byte[] pwdPadding=new byte[16];
			int iLen=16;
			if(password.length<16)
				iLen=password.length;
			System.arraycopy(password, 0, pwdPadding, 0, iLen);
			for(;iLen<16;++iLen)
				pwdPadding[iLen]=(byte)(0x2a+iLen);
			
			//对密码进行转换
			byte[] pwdXor=null;
			if(password.length<20) {
				iLen=0;
				pwdXor=new byte[] {0x54, (byte)0xCA, (byte)0x9C,0x3D};
			}else if(password.length<=32) {
				iLen=password.length-16;
			}else {
				iLen=16;
			}
			if(iLen>0) {
				pwdXor=new byte[iLen];
				System.arraycopy(password, 16, pwdXor, 0, iLen);
			}
			for(iLen=0;iLen<pwdXor.length;++iLen) {
				pwdPadding[iLen]^=pwdXor[iLen];
			}

			SecretKeySpec key = new SecretKeySpec(pwdPadding, "AES");
			Cipher cipher = Cipher.getInstance("AES");// 创建密码器
			if(encrypt)// 初始化ENCRYPT_MODE 加密
				cipher.init(Cipher.ENCRYPT_MODE, key);
			else
				cipher.init(Cipher.DECRYPT_MODE, key);
			return cipher.doFinal(input);
		}catch(Exception ex){
			return null;
		}
	}

}
