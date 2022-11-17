package demo;

import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * @author 吴嘉烺
 * @description
 * @date 2022/10/12
 */
@Slf4j
public class LockDemo {


    /**
     * 不涉及共享资源的慢方法
     */
    private void slow() {
        try {
            TimeUnit.MILLISECONDS.sleep(10);
        } catch (InterruptedException e) {
        }
    }

    public int wrong() {
        List<Integer> data = new ArrayList<>();
        long begin = System.currentTimeMillis();
        IntStream.rangeClosed(1, 1000).parallel().forEach(i -> {
            //加锁粒度太粗了
            synchronized (this) {
                slow();
                data.add(i);
            }
        });
        log.info("took:{}", System.currentTimeMillis() - begin);
        return data.size();
    }

    public int right() {
        List<Integer> data = new ArrayList<>();
        long begin = System.currentTimeMillis();
        IntStream.rangeClosed(1, 1000).parallel().forEach(i -> {
            slow();
            //只对List加锁
            synchronized (data) {
                data.add(i);
            }
        });
        log.info("took:{}", System.currentTimeMillis() - begin);
        return data.size();
    }

    public int normal() {
        List<Integer> data = new ArrayList<>();
        long begin = System.currentTimeMillis();
        IntStream.rangeClosed(1, 1000).forEach(i -> {
            slow();
            data.add(i);
        });
        log.info("took:{}", System.currentTimeMillis() - begin);
        return data.size();
    }

    public int concurrent() {
        ConcurrentLinkedQueue data = new ConcurrentLinkedQueue();
        long begin = System.currentTimeMillis();
        IntStream.rangeClosed(1, 1000).parallel().forEach(i -> {
            slow();
            data.add(i);
        });
        log.info("took:{}", System.currentTimeMillis() - begin);
        return data.size();
    }

    public static void main(String[] args) {
        LockDemo lockDemo = new LockDemo();
        System.out.println(lockDemo.wrong());
        System.out.println(lockDemo.right());
        System.out.println(lockDemo.normal());
        System.out.println(lockDemo.concurrent());
    }
}
