package demo;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * @author 吴嘉烺
 * @description
 * @date 2022/10/13
 */
@Slf4j
public class ThreadPoolExecutorDemo {

    private void printStats(ThreadPoolExecutor threadPool) {

        Executors.newSingleThreadScheduledExecutor().scheduleAtFixedRate(() -> {

            log.info("=========================");
            log.info("Pool Size: {}", threadPool.getPoolSize());
            log.info("Active Threads: {}", threadPool.getActiveCount());
            log.info("Number of Tasks Completed: {}", threadPool.getCompletedTaskCount());
            log.info("Number of Tasks in Queue: {}", threadPool.getQueue().size());
            log.info("=========================");

        }, 0, 1, TimeUnit.SECONDS);

    }

    public void test() throws InterruptedException {
        //使用一个计数器跟踪完成的任务数
        AtomicInteger atomicInteger = new AtomicInteger();

        //创建一个具有2个核心线程、5个最大线程，使用容量为10的ArrayBlockingQueue阻塞队列作为工作队列的线程池，使用默认的AbortPolicy拒绝策略
        ThreadPoolExecutor threadPool = new ThreadPoolExecutor(
                2, 5,
                5, TimeUnit.SECONDS,
                new ArrayBlockingQueue<>(10),
                new ThreadFactoryBuilder().setNameFormat("demo-wjlPool-%d").build(),
                new ThreadPoolExecutor.AbortPolicy());

        printStats(threadPool);

        //每隔1秒提交一次，一共提交20次任务
        IntStream.rangeClosed(1, 20).forEach(i -> {

            try {
                TimeUnit.SECONDS.sleep(1);

            } catch (InterruptedException e) {
                e.printStackTrace();

            }

            int id = atomicInteger.incrementAndGet();

            try {
                threadPool.submit(() -> {
                    log.info("{} started", id);
                    log.info("{} started", Thread.currentThread().getName());
                    //每个任务耗时10秒
                    try {
                        TimeUnit.SECONDS.sleep(10);
                    } catch (InterruptedException e) {

                    }
                    log.info("{} finished", id);
                });

            } catch (Exception ex) {

                //提交出现异常的话，打印出错信息并为计数器减一
                log.error("error submitting task {}", id, ex);
                atomicInteger.decrementAndGet();

            }

        });

        TimeUnit.SECONDS.sleep(60);
        System.out.println(atomicInteger.intValue());
    }

    public static void main(String[] args) throws InterruptedException {
        ThreadPoolExecutorDemo threadPoolExecutorDemo = new ThreadPoolExecutorDemo();
        threadPoolExecutorDemo.test();
    }

}
