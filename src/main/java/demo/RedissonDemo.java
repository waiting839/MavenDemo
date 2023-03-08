package demo;

import org.redisson.Redisson;
import org.redisson.api.RLock;
import org.redisson.api.RedissonClient;
import org.redisson.config.Config;

import java.util.concurrent.CompletableFuture;

/**
 * @author 吴嘉烺
 * @description
 * @date 2023/3/8
 */
public class RedissonDemo {
    public static void main(String[] args) throws Exception {
        Config config = new Config();
        config.useSingleServer().setAddress("redis://127.0.0.1:6379");
        RedissonClient redissonClient = Redisson.create(config);
        if (redissonClient == null) {
            System.out.println("连接redis失败");
        } else {
            System.out.println("连接redis成功。 redissonClient：" + redissonClient);
            RLock rLock = redissonClient.getLock("lock");
            CompletableFuture<Void> task1 = CompletableFuture.supplyAsync(() -> {
                rLock.lock();
                try {
                    System.out.println("task1加锁成功，执行后续代码。线程 ID：" + Thread.currentThread().getId());
                    Thread.sleep(3000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    rLock.unlock();
                    System.out.println("Finally，task1释放锁成功。线程 ID：" + Thread.currentThread().getId());
                }
                return null;
            });
            CompletableFuture<Void> task2 = CompletableFuture.supplyAsync(() -> {
                rLock.lock();
                try {
                    System.out.println("task2加锁成功，执行后续代码。线程 ID：" + Thread.currentThread().getId());
                    Thread.sleep(3000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    rLock.unlock();
                    System.out.println("Finally，task2释放锁成功。线程 ID：" + Thread.currentThread().getId());
                }
                return null;
            });
            task1.get();
            task2.get();

        }
    }
}
