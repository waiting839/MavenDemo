package demo;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * @author 作者 吴嘉烺
 * @description 类描述
 * @date 创建时间 2022/1/18
 */
public class CompletableFutureDemo {

    final static ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(10, 10, 10, TimeUnit.SECONDS, new LinkedBlockingQueue<>());

    public static void main(String[] args) throws Exception{
        CompletableFuture<Student> completableFuture = CompletableFuture.supplyAsync(() -> {
            Student student = new Student();
            System.out.println(Thread.currentThread().getName());
            return student;
        }).thenApply(s -> {
            s.setName("666");
            System.out.println(Thread.currentThread().getName());
            String str = null;
//            str.split(",");
            return s;
        }).whenComplete((s, ex) -> {
            System.out.println(Thread.currentThread().getName());
            if(ex != null){
                System.out.println(ex.getMessage());
            }
        });
        Student student = completableFuture.get();
        System.out.println(student.getName());
        Student studentTest = new Student();
        CompletableFuture<Student> task1 = CompletableFuture.supplyAsync(() -> {
            studentTest.setName("test");
            System.out.println(Thread.currentThread().getName());
            return studentTest;
        }, threadPoolExecutor);
        CompletableFuture<Student> task2 = CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            studentTest.setAge(9);
            System.out.println(Thread.currentThread().getName());
            String str = null;
            return studentTest;
        }, threadPoolExecutor);
        CompletableFuture<Void> future = CompletableFuture.allOf(task1, task2);
        try {
            future.join();
        }catch (Exception e){
            throw e;
        }finally {
            threadPoolExecutor.shutdown();
        }
        System.out.println(studentTest.toString());
    }
}
