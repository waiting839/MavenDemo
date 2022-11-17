package demo;

import net.sf.ehcache.Cache;
import net.sf.ehcache.CacheManager;
import net.sf.ehcache.Element;
import redis.clients.jedis.Jedis;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import redis.clients.jedis.Pipeline;
import redis.clients.jedis.Response;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author 作者 吴嘉烺
 * @description 类描述
 * @date 创建时间 2021/10/8
 */
public class CacheDemo {

    private Jedis jedis;

    public CacheDemo(Jedis jedis){
        this.jedis = jedis;
    }

    public boolean isActionAllowed(String userId, String actionKey, int period, int maxCount) throws IOException {
        String key = String.format("hist:%s:%s", userId, actionKey);
        long nowTs = System.currentTimeMillis();
        Pipeline pipeline = jedis.pipelined();
        //设置原子性
        pipeline.multi();
        pipeline.zadd(key, nowTs, "" + nowTs);
        //移除有序集中，指定分数（score）区间内的所有成员。
        pipeline.zremrangeByScore(key, 0, nowTs - period * 1000);
        Response<Long> count = pipeline.zcard(key);
        pipeline.expire(key, period + 1);
        pipeline.exec();
        pipeline.close();
        System.out.println("当前count值：" + count.get());
        return count.get() <= maxCount;
    }

    public static void main(String[] args) throws IOException {
        //连接本地的 Redis 服务
        Jedis jedis = new Jedis("localhost");
        System.out.println("连接成功");
//        jedis.set("a", "1");
//        jedis.set("b", "2");
        //设置 redis 字符串数据
//        Map<String, String> map = new HashMap<>();
//        map.put("1", "一");
//        map.put("2", "二");
//        jedis.set("map", map.toString());
//        // 获取存储的数据并输出
//        System.out.println("redis 存储的字符串为: "+ jedis.get("map"));
//        System.out.println("修改map");
//        map.remove("1");
//        System.out.println("redis 存储的字符串为: "+ jedis.get("map"));

        //分布式锁
//        String res = jedis.set("lock", "true", "NX", "EX", 50);
//        System.out.println(res);

//        for(int i = 0; i < 1000; i++){
//            jedis.pfadd("codehole", "user" + i);
//
//        }
//        long total = jedis.pfcount("codehole");
//        System.out.println((total) + "/" + (1000));

//        // 1. 创建缓存管理器
//        CacheManager cacheManager = CacheManager.create("./src/main/resources/ehcache");
//
//        // 2. 获取缓存对象
//        Cache cache = cacheManager.getCache("HelloWorldCache");
//
//        // 3. 创建元素
//        List<String> list = new ArrayList<>();
//        list.add("1");
//        list.add("2");
//        list.add("3");
//        System.out.println("新建list：");
//        list.stream().forEach(e -> System.out.print(e + " "));
//        Element element = new Element("list", list);
//
//        System.out.println();
//        System.out.println("加入缓存");
//        // 4. 将元素添加到缓存
//        cache.put(element);
//
//        // 5. 获取缓存
//        Element value = cache.get("list");
//
//        List<String> cacheList = (List<String>)value.getObjectValue();
//        System.out.println();
//        System.out.println("获取list: ");
//        cacheList.stream().forEach(e -> System.out.print(e + " "));
//
//        cacheList.remove(1);
//        System.out.println();
//        System.out.println("删除一个元素，list：");
//        cacheList.stream().forEach(e -> System.out.print(e + " "));
//
//        System.out.println();
//        System.out.println("再次获取缓存");
//        Element value2 = cache.get("list");
//        List<String> cacheList2 = (List<String>)value2.getObjectValue();
//        System.out.println();
//        System.out.println("再次获取list：");
//        cacheList2.stream().forEach(e -> System.out.print(e + " "));
//        cacheManager.shutdown();

//        CacheDemo cacheDemo = new CacheDemo(jedis);
//        for(int i = 0; i < 20; i++){
//            System.out.println(cacheDemo.isActionAllowed("xiaowu", "reply", 60, 5 ));
//        }
//        jedis.close();
    }
}
