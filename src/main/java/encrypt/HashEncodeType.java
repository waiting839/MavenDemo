package encrypt;

/**
 * 转摘要信息的已编码类型。本系统中，做了两层处理：sha1+md5
 * @author dnnyo
 *
 */
public enum HashEncodeType
{
	/**
	 * 原始内容
	 */
	original,
	/**
	 * sha1编码
	 */
	sha1,
	/**
	 * md5编码
	 */
	md5
}
