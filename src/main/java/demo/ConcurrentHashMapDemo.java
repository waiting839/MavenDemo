package demo;

import lombok.extern.slf4j.Slf4j;
import org.springframework.util.Assert;
import org.springframework.util.StopWatch;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

/**
 * @author 吴嘉烺
 * @description
 * @date 2022/10/11
 */
@Slf4j
public class ConcurrentHashMapDemo {

    //线程个数
    private static int THREAD_COUNT = 10;
    //总元素数量
    private static int ITEM_COUNT_1000 = 1000;
    //总元素数量
    private static int ITEM_COUNT_10 = 10;
    //循环次数
    private static int LOOP_COUNT = 10000000;

    /**
     * 帮助方法，用来获得一个指定元素数量模拟数据的ConcurrentHashMap
     * @param count
     * @return
     */
    private static ConcurrentHashMap<String, Long> getData(int count) {
        return LongStream.rangeClosed(1, count)
                .boxed()
                .collect(Collectors.toConcurrentMap(i -> UUID.randomUUID().toString(), Function.identity(),
                        (o1, o2) -> o1, ConcurrentHashMap::new));
    }

    public void addData() throws InterruptedException {
        ConcurrentHashMap<String, Long> concurrentHashMap = getData(ITEM_COUNT_1000 - 100);
        //初始900个元素
        log.info("init size:{}", concurrentHashMap.size());
        ForkJoinPool forkJoinPool = new ForkJoinPool(THREAD_COUNT);
        //使用线程池并发处理逻辑
        forkJoinPool.execute(() -> IntStream.rangeClosed(1, 10).parallel().forEach(i -> {
            //查询还需要补充多少个元素
            synchronized (concurrentHashMap){
                int gap = ITEM_COUNT_1000 - concurrentHashMap.size();
                log.info("gap size:{}", gap);
                //补充元素
                concurrentHashMap.putAll(getData(gap));
            }
        }));
        //等待所有任务完成
        forkJoinPool.shutdown();
        forkJoinPool.awaitTermination(1, TimeUnit.HOURS);
        //最后元素个数会是1000吗？
        log.info("finish size:{}", concurrentHashMap.size());
    }

    public Map<String, Long> normalUse() throws InterruptedException {
        ConcurrentHashMap<String, Long> freqs = new ConcurrentHashMap<>(ITEM_COUNT_10);
        ForkJoinPool forkJoinPool = new ForkJoinPool(THREAD_COUNT);
        forkJoinPool.execute(() -> IntStream.rangeClosed(1, LOOP_COUNT).parallel().forEach(i -> {
                    //获得一个随机的Key
                    String key = "item" + ThreadLocalRandom.current().nextInt(ITEM_COUNT_10);
                    synchronized (freqs) {
                        if (freqs.containsKey(key)) {
                            //Key存在则+1
                            freqs.put(key, freqs.get(key) + 1);
                        } else {
                            //Key不存在则初始化为1
                            freqs.put(key, 1L);
                        }
                    }
                }
        ));
        forkJoinPool.shutdown();
        forkJoinPool.awaitTermination(1, TimeUnit.HOURS);
        return freqs;
    }

    public Map<String, Long> goodUse() throws InterruptedException {
        ConcurrentHashMap<String, LongAdder> freqs = new ConcurrentHashMap<>(ITEM_COUNT_10);
        ForkJoinPool forkJoinPool = new ForkJoinPool(THREAD_COUNT);
        forkJoinPool.execute(() -> IntStream.rangeClosed(1, LOOP_COUNT).parallel().forEach(i -> {
                    //获得一个随机的Key
                    String key = "item" + ThreadLocalRandom.current().nextInt(ITEM_COUNT_10);
                    freqs.computeIfAbsent(key, v -> new LongAdder()).increment();
                }
        ));
        forkJoinPool.shutdown();
        forkJoinPool.awaitTermination(1, TimeUnit.HOURS);
        return freqs.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue().longValue()));
    }

    public static void main(String[] args) throws InterruptedException {
        ConcurrentHashMapDemo concurrentHashMapDemo = new ConcurrentHashMapDemo();
//        concurrentHashMapDemo.addData();
        StopWatch stopWatch = new StopWatch();
        stopWatch.start("normalUse");
        Map<String, Long> normalUse = concurrentHashMapDemo.normalUse();
        stopWatch.stop();
        //校验元素数量
        Assert.isTrue(normalUse.size() == ITEM_COUNT_10, "normalUse size error");
        //校验累计总数
        Assert.isTrue(normalUse.entrySet().stream()
                        .mapToLong(item -> item.getValue()).reduce(0, Long::sum) == LOOP_COUNT
                , "normalUse count error");
        stopWatch.start("goodUse");
        Map<String, Long> goodUse = concurrentHashMapDemo.goodUse();
        stopWatch.stop();
        //校验元素数量
        Assert.isTrue(goodUse.size() == ITEM_COUNT_10, "goodUse size error");
        //校验累计总数
        Assert.isTrue(goodUse.entrySet().stream()
                        .mapToLong(item -> item.getValue()).reduce(0, Long::sum) == LOOP_COUNT
                , "goodUse count error");
        log.info(stopWatch.prettyPrint());
    }
}
