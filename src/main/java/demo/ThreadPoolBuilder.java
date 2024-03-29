package demo;

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * @author 吴嘉烺
 * @description
 * @date 2022/10/14
 */
@Slf4j
public class ThreadPoolBuilder {
    private static ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(
            10, 50,
            2, TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(1000),
            new ThreadFactoryBuilder().setNameFormat("demo-wjlPool-%d")
                    .setUncaughtExceptionHandler((thread, throwable) -> log.error("ThreadPool {} got exception", thread, throwable))
                    .build());

    private static ForkJoinPool forkJoinPool = new ForkJoinPool(30);

    public static ThreadPoolExecutor getRightThreadPool() {
        return threadPoolExecutor;
    }

    public static ForkJoinPool getForkJoinPool() {
        return forkJoinPool;
    }

}
