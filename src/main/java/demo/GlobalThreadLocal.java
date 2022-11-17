package demo;


import java.util.HashMap;

/**
 * 用于过滤器设置的全局与当前线程相关的变量。
 * 并在GlobalThreadLocalFilter中清理。
 * @author dennis
 *
 */
public class GlobalThreadLocal {

	/**其他需要设置的线程变量，通过key区别不同的变量*/
	private static final ThreadLocal<HashMap<String,Object>> THREAD_MAP_VAR = new ThreadLocal<>();


	/**
	 * 清理本地线程数据。
	 */
	public static void clear()
	{
		THREAD_MAP_VAR.remove();
	}

	/**
	 * 获取线程对象中保存的某个变量值
	 * @param key 变量值
	 * @return 返回结果
	 */
	public static Object getMapValue(String key)
	{
		HashMap<String,Object> mapVar=THREAD_MAP_VAR.get();
		if(mapVar!=null) {
			return mapVar.get(key);
		}
		return null;
	}
	
	/**
	 * 
	 * @param key
	 * @return
	 */
	public static void setMapValue(String key,Object value)
	{
		HashMap<String,Object> mapVar=THREAD_MAP_VAR.get();
		if(mapVar==null) {
			mapVar=new HashMap<>();
			THREAD_MAP_VAR.set(mapVar);
		}
		
		mapVar.put(key, value);
	}



}
