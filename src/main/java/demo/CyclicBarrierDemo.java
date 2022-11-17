package demo;

import java.util.Map;
import java.util.concurrent.*;

/**
 * @author 作者 吴嘉烺
 * @description 类描述
 * @date 创建时间 2021/9/3
 */
public class CyclicBarrierDemo {

    private static Map<String, String> map = new ConcurrentHashMap<>();

    private static final Integer threadCount = 10;

    //自定义回调函数
    private static final CyclicBarrier cyclicBarrier = new CyclicBarrier(5, () -> {
        printTest(map.size());
    });

    private static final CountDownLatch countDownLatch = new CountDownLatch(2);

    public static void cyclicBarrierTest(int threadNum){
        map.put("threadNum " + threadNum, "is ready");
        System.out.println("threadNum " + threadNum + " : is ready");
        try {
            String str = null;
            str.split(",");
            cyclicBarrier.await(2, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (BrokenBarrierException e) {
            e.printStackTrace();
        } catch (TimeoutException e) {
            e.printStackTrace();
        }
        System.out.println("threadNum " + threadNum + " : is finish");
    }

    public static void printTest(int size){
        System.out.println("------------IS READY------------AND MAP.SIZE : " + size);
    }

    public static void main(String[] args) throws Exception{
//        Thread.setDefaultUncaughtExceptionHandler(new Thread.UncaughtExceptionHandler(){
//            @Override
//            public void uncaughtException(Thread t, Throwable e) {
//                System.out.println("caugth: " + e);
//            }
//        });
        ThreadPoolExecutor threadPool = new ThreadPoolExecutor(2,4,5,
                TimeUnit.SECONDS,new ArrayBlockingQueue<>(9),new ThreadPoolExecutor.DiscardOldestPolicy()){
            /**
             *
             * @param t   执行任务的线程
             * @param r 将要被执行的任务
             */
            @Override
            protected void afterExecute(Runnable r, Throwable t) {
                System.out.println("=============出错了=============");
                t.printStackTrace();
            }
        };
        for(int i = 0; i < threadCount; i++){
            final Integer threadNum = i;
            Thread.sleep(1000);
            threadPool.execute(() -> {
                cyclicBarrierTest(threadNum);
            });
        }
        threadPool.shutdown();
    }

}
