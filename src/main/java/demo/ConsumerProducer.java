package demo;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * @author 吴嘉烺
 * @description
 * @date 2023/7/20
 */
public class ConsumerProducer {
    public static final String EXIT_MSG = "Good bye!";

    public static void main(String[] args) {
        BlockingQueue<String> queue = new ArrayBlockingQueue<>(3);
        Producer producer = new Producer(queue);
        Consumer consumer = new Consumer(queue);
        new Thread(producer).start();
        new Thread(consumer).start();
    }

    static class Producer implements Runnable {
        private BlockingQueue<String> queue;
        public Producer(BlockingQueue<String> queue) {
            this.queue = queue;
        }
        @Override
        public void run() {
            for (int i = 0; i < 10; i++) {
                try {
                    Thread.sleep(500L);
                    String msg = "Message" + i;
                    System.out.println("Produced new item: " + msg);
                    queue.put(msg);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            try {
                System.out.println("Time to say good bye!");
                queue.put(EXIT_MSG);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    static class Consumer implements Runnable {
        private BlockingQueue<String> queue;
        public Consumer(BlockingQueue<String> queue) {
            this.queue = queue;
        }
        @Override
        public void run() {
            try {
                String msg;
                while (!EXIT_MSG.equalsIgnoreCase((msg = queue.take()))) {
                    System.out.println("Consumed item: " + msg);
                    Thread.sleep(1000L);
                }
                System.out.println("Got exit message, bye!");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
