package demo;

import java.util.Arrays;
import java.util.List;

/**
 * @author 作者 吴嘉烺
 * @description 类描述
 * @date 创建时间 2021/9/7
 */
public class ThreadLocalDemo {

    public static void main(String[] args) {
        GlobalThreadLocal.setMapValue("666", "666");
        List<String> list = Arrays.asList("aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg");
        list.parallelStream().forEach(v -> {
            System.out.println(Thread.currentThread().getName() + " " + v + " threadLocal: " + GlobalThreadLocal.getMapValue("666"));
        });
    }
}
