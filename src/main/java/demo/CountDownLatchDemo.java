package demo;

import java.util.concurrent.*;

/**
 * @author 作者 吴嘉烺
 * @description 类描述
 * @date 创建时间 2021/9/7
 */
public class CountDownLatchDemo {

    public void errorTest() throws Exception {
        ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(2, 3, 5, TimeUnit.SECONDS, new ArrayBlockingQueue<>(5));
        final CountDownLatch countDownLatch = new CountDownLatch(3);
        ConcurrentLinkedQueue<Exception> exceptions = new ConcurrentLinkedQueue<>();
        for(int i = 0; i < 3; i++){
            threadPoolExecutor.execute(() -> {
                try{
                    String str = null;
                    str.split(",");
                }catch (Exception e){
                    exceptions.add(e);
                    throw new IndexOutOfBoundsException(e.getMessage());
                }finally {
                    countDownLatch.countDown();
                }
            });
        }
        countDownLatch.await();
        threadPoolExecutor.shutdown();
        if(!exceptions.isEmpty()){
            throw new Exception(exceptions.poll());
        }
    }

    public void test(Person person){
        ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(2, 3, 5, TimeUnit.SECONDS, new ArrayBlockingQueue<>(5));
        final CountDownLatch countDownLatch = new CountDownLatch(3);
        ConcurrentLinkedQueue<Exception> exceptions = new ConcurrentLinkedQueue<>();

    }

    public static void main(String[] args) throws Exception {
//        CountDownLatchDemo countDownLatchDemo = new CountDownLatchDemo();
//        countDownLatchDemo.errorTest();
        ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(2, 3, 5, TimeUnit.SECONDS, new ArrayBlockingQueue<>(5));
        final CountDownLatch countDownLatch = new CountDownLatch(2);
        Person person = new Person();
        threadPoolExecutor.execute(() -> {
            person.setName("xiaowu");
            countDownLatch.countDown();
        });
        threadPoolExecutor.execute(() -> {
            person.setAge(18);
            countDownLatch.countDown();
        });
        countDownLatch.await();
        System.out.println(person.toString());
        threadPoolExecutor.shutdown();
    }
}
