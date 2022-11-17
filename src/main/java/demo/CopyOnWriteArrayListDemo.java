package demo;

import lombok.extern.slf4j.Slf4j;
import org.springframework.util.StopWatch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author 吴嘉烺
 * @description
 * @date 2022/10/11
 */
@Slf4j
public class CopyOnWriteArrayListDemo {
    /**
     * 帮助方法用来填充List
     * @param list
     */
    private void addAll(List<Integer> list) {
        list.addAll(IntStream.rangeClosed(1, 1000000).boxed().collect(Collectors.toList()));
    }

    public void testRead(){
        //创建两个测试对象
        List<Integer> copyOnWriteArrayList = new CopyOnWriteArrayList<>();
        List<Integer> synchronizedList = Collections.synchronizedList(new ArrayList<>());
        List<Integer> list = new ArrayList<>();
        //填充数据
        addAll(copyOnWriteArrayList);
        addAll(synchronizedList);
        addAll(list);
        StopWatch stopWatch = new StopWatch();
        int loopCount = 1000000;
        int count = copyOnWriteArrayList.size();
        stopWatch.start("copyOnWriteArrayList");
        IntStream.rangeClosed(1, loopCount).parallel().forEach(__ -> copyOnWriteArrayList.get(ThreadLocalRandom.current().nextInt(count)));
        stopWatch.stop();
        stopWatch.start("synchronizedList");
        IntStream.rangeClosed(1, loopCount).parallel().forEach(__ -> synchronizedList.get(ThreadLocalRandom.current().nextInt(count)));
        stopWatch.stop();
        stopWatch.start("list");
        IntStream.rangeClosed(1, loopCount).parallel().forEach(__ -> list.get(ThreadLocalRandom.current().nextInt(count)));
        stopWatch.stop();
        log.info(stopWatch.prettyPrint());
    }

    public static void main(String[] args) {
        CopyOnWriteArrayListDemo copyOnWriteArrayListDemo = new CopyOnWriteArrayListDemo();
        copyOnWriteArrayListDemo.testRead();
    }
}
