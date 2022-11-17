package demo;

import lombok.Getter;
import sun.misc.Unsafe;

import java.lang.reflect.Field;

/**
 * @author 吴嘉烺
 * @description
 * @date 2022/9/8
 */
@Getter
public class UnsafeDemo implements Runnable{

    /*volatile*/ boolean flag = false;

    @Override
    public void run() {
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println("subThread change flag to:" + flag);
        flag = true;
    }

    private static Unsafe reflectGetUnsafe() {
        try {
            Field field = Unsafe.class.getDeclaredField("theUnsafe");
            field.setAccessible(true);
            return (Unsafe) field.get(null);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static void main(String[] args) throws InterruptedException {
        UnsafeDemo unsafeDemo = new UnsafeDemo();
        Unsafe unsafe = UnsafeDemo.reflectGetUnsafe();
        new Thread(unsafeDemo).start();
        long startTime = System.currentTimeMillis();
        while (true) {
            boolean flag = unsafeDemo.isFlag();
            unsafe.loadFence();
            if(flag){
                System.out.println("detected flag changed");
                break;
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.println("main thread end");
        System.out.println("用时：" + (endTime - startTime));
    }
}
