package demo;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import lombok.Data;

import com.google.common.collect.Lists;
import feign.FeignException;
import org.apache.commons.lang3.StringUtils;

/**
 * @author 作者 吴嘉烺
 * @description 类描述
 * @date 创建时间 2021/8/23
 */
@Data
public class Demo {

    boolean flag = false;

    private static final ExecutorService FORM_EXECUTOR = new ThreadPoolExecutor(300, 600,
            10L, TimeUnit.SECONDS, new LinkedBlockingQueue<>(1200));

    public void test() {
        List<Integer> list = new ArrayList<>();
        for(int i = 0; i < 10000000; i++){
            list.add((int)(Math.random()*1000000));
        }
//        for(Integer num : list){
//            System.out.print(num + " ");
//        }
//        ForkJoinPool forkJoinPool = new ForkJoinPool(10);
//        long start = System.currentTimeMillis();
//        List<Integer> newList = forkJoinPool.submit(() -> list.parallelStream()
//                .filter(e -> e > 30000)
//                .collect(Collectors.toList())
//        ).join();
//        long end = System.currentTimeMillis();
//        System.out.println();
//        System.out.println("时间：" + (end - start));
        long start2 = System.currentTimeMillis();
        List<Integer> newList2 = list.stream()
                .filter(e -> e > 30000)
                .collect(Collectors.toList());
        long end2 = System.currentTimeMillis();
        System.out.println();
        System.out.println("时间：" + (end2 - start2));
//        for(Integer num : newList){
//            System.out.print(num + " ");
//        }
    }

    public static boolean equalss(String s1, String s2) {
        if (s2 instanceof String) {
            int n = s1.length();
            if (n == s2.length()) {
                char v1[] = s1.toCharArray();
                char v2[] = s2.toCharArray();
                int i = 0;
                while (n-- != 0) {
                    if (v1[i] != v2[i]) {
                        System.out.println("第 " + i + "个字节不一样");
                        System.out.println(s1.substring(0, i + 5));
                        System.out.println(s2.substring(0, i + 5));
                        return false;
                    }
                    i++;
                }
                return true;
            }
        }
        return false;
    }

    public static List<Student> tttt(List<String> list, List<Student> studentList){
        List<Student> result = studentList.stream().filter(e -> list.contains(e.getName())).collect(Collectors.toList());
        System.out.println(result.size());
        return result;
    }

    private String getFuzzyQueryStr(String str){
        StringBuffer stringBuffer = new StringBuffer();
        for(char c : str.toCharArray()){
            if(c == '%' || c == '_'){
                stringBuffer.append("\\" + c);
            }else {
                stringBuffer.append(c);
            }
        }
        return stringBuffer.toString();
    }

    public static boolean isNullorEmpty(String value)
    {
        return value == null || value.isEmpty();
    }

    public static int byteToUnsignedInt(byte data) {
        return data & 0xff;
    }

    public static boolean isUTF8(byte[] pBuffer) {
        boolean IsUTF8 = true;
        boolean IsASCII = true;
        int size = pBuffer.length;
        int i = 0;
        while (i < size) {
            int value = byteToUnsignedInt(pBuffer[i]);
            if (value < 0x80) {
                // (10000000): 值小于 0x80 的为 ASCII 字符
                if (i >= size - 1) {
                    if (IsASCII) {
                        // 假设纯 ASCII 字符不是 UTF 格式
                        IsUTF8 = false;
                    }
                    break;
                }
                i++;
            } else if (value < 0xC0) {
                // (11000000): 值介于 0x80 与 0xC0 之间的为无效 UTF-8 字符
                IsASCII = false;
                IsUTF8 = false;
                break;
            } else if (value < 0xE0) {
                // (11100000): 此范围内为 2 字节 UTF-8 字符
                IsASCII = false;
                if (i >= size - 1) {
                    break;
                }

                int value1 = byteToUnsignedInt(pBuffer[i + 1]);
                if ((value1 & (0xC0)) != 0x80) {
                    IsUTF8 = false;
                    break;
                }

                i += 2;
            } else if (value < 0xF0) {
                IsASCII = false;
                // (11110000): 此范围内为 3 字节 UTF-8 字符
                if (i >= size - 2) {
                    break;
                }

                int value1 = byteToUnsignedInt(pBuffer[i + 1]);
                int value2 = byteToUnsignedInt(pBuffer[i + 2]);
                if ((value1 & (0xC0)) != 0x80 || (value2 & (0xC0)) != 0x80) {
                    IsUTF8 = false;
                    break;
                }

                i += 3;
            }  else if (value < 0xF8) {
                IsASCII = false;
                // (11111000): 此范围内为 4 字节 UTF-8 字符
                if (i >= size - 3) {
                    break;
                }

                int value1 = byteToUnsignedInt(pBuffer[i + 1]);
                int value2 = byteToUnsignedInt(pBuffer[i + 2]);
                int value3 = byteToUnsignedInt(pBuffer[i + 3]);
                if ((value1 & (0xC0)) != 0x80
                        || (value2 & (0xC0)) != 0x80
                        || (value3 & (0xC0)) != 0x80) {
                    IsUTF8 = false;
                    break;
                }

                i += 3;
            } else {
                IsUTF8 = false;
                IsASCII = false;
                break;
            }
        }

        return IsUTF8;
    }

    private Integer getInteger() {
        Integer integer = new Integer(127);
        return integer;
    }

    public static void main(String[] args) throws Throwable {
//        List<String> list = Arrays.asList("m,k,l,a", "1,3,5,7");
//        System.out.println(list);
//        List<String> newList = list.stream().flatMap(s -> {
//            String[] split = s.split(",");
//            Stream<String> stream = Arrays.stream(split);
//            return stream;
//        }).collect(Collectors.toList());
//        System.out.println(newList);
//        Demo demo = new Demo();
//        demo.test();

//        final List<String> list = new ArrayList<>(Arrays.asList("m,k,l,a", "null"));
//        list.add("1,2,3,5");
//        list.parallelStream().forEach(v -> {
//            String[] s = v.split(",");
//            System.out.println(s[0]);
//        });

//        List<Student> list = new ArrayList<>();
//        Student stu1 = new Student("stu1", 10);
//        Student stu2 = new Student("stu2", 20);
//        list.add(stu1);
//        list.add(stu2);
//        List<Student> copyList = new ArrayList<>();
//        CollectionUtils.addAll(copyList, new Object[list.size()]);
//        Collections.copy(copyList, list);
//        list.get(0).setName("stu666");
//        list.stream().forEach(e -> System.out.println(e.getName() + " " + e.getAge()));
//        copyList.stream().forEach(e -> System.out.println(e.getName() + " " + e.getAge()));
//        copyList.addAll(list);
//        list.stream().forEach(e -> System.out.println(e.getName() + " " + e.getAge()));
//        copyList.stream().forEach(e -> System.out.println(e.getName() + " " + e.getAge()));
//        list.get(0).setName("stu666");
//        list.stream().forEach(e -> System.out.println(e.getName() + " " + e.getAge()));
//        copyList.stream().forEach(e -> System.out.println(e.getName() + " " + e.getAge()));
//        list.remove(0);
//        list.stream().forEach(e -> System.out.println(e.getName() + " " + e.getAge()));
//        copyList.stream().forEach(e -> System.out.println(e.getName() + " " + e.getAge()));

//        List<Map<String, String>> mapList = new ArrayList<>();
//        HashMap<String, String> map1 = new HashMap<>(16);
//        map1.put("111","一一一");
//        map1.put("222","二二二");
//        map1.put("333","三三三");
//        mapList.add(map1);
//        HashMap<String, String> map2 = new HashMap<>(16);
//        map2.put("1111","一一一一");
//        map2.put("2222","二二二二");
//        map2.put("3333","三三三三");
//        mapList.add(map2);
//        boolean flag = false;
//        System.out.println("123" + flag);

//        list.size();
//        String fieldsStr = StringUtils.strip(list.toString(),"[]").replace(" ","");
//        System.out.println(fieldsStr);

//        List<String> list = new ArrayList<>(Arrays.asList(new String[]{"210802", "210731", "210729", "20210715", "20210713", "20210721"}));
//        Iterator<String> iterator = list.listIterator();
//        while(iterator.hasNext()){
//            String str = iterator.next();
//            if(str.equals("210731")){
//                iterator.remove();
//            }
//            System.out.println(list.toString());
//        }
//        List<String> linkedFormIds = new ArrayList<>();
//        Student student = new Student();
//        CompletableFuture<Object> future = CompletableFuture.supplyAsync(() -> {
//            //TODO
//            System.out.println("666");
//            if(1 == 1){
//                student.setAge(2);
//                return student;
//            }
//            List<String> linkedFormList = new ArrayList<>();
//            linkedFormList.add("eee");
//            try {
//                Thread.sleep(1000);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//            return linkedFormList;
//        });
//        GlobalThreadLocal.setMapValue("future", future);
//        Thread.sleep(500);
//        CompletableFuture<Object> future2 = (CompletableFuture<Object>) GlobalThreadLocal.getMapValue("future");
//        Student s = (Student) future2.get(1500, TimeUnit.MILLISECONDS);
//        System.out.println(s.getAge());

//        List<String> list = new LinkedList<>();
//        list.stream().forEach(e -> {
//            System.out.println("66:" + e.split(","));
//        });

//        LocalDateTime localDateTime = LocalDateTime.now().withNano(0);
//        System.out.println(localDateTime);

//        Student stu1 = new Student("stu1", 10);
//        Student stu2 = new Student("stu2", 20);
//        Student stu3 = new Student("stu2", 30);
//        Student stu4 = new Student();
//        List<Student> list = new LinkedList<>();
//        list.add(stu1);
//        list.add(stu2);
//        list.add(stu3);
//        list.add(stu4);
//        list = list.stream().filter(e -> (e.getName() != null)).collect(
//                collectingAndThen(
//                        toCollection(() -> new TreeSet<>(Comparator.comparing(Student::getName))), ArrayList::new
//                )
//        );
//        Map<String, Object> map = list.stream().collect(Collectors.toMap(Student::getName, e -> e,  (a, b) -> b, HashMap::new));
//        map.entrySet().stream().forEach(e -> {
//            System.out.println(e.getKey() + " : " + e.getValue().toString());
//        });

//        Integer a = null;
//        Optional.ofNullable(a).orElseThrow(RuntimeException::new);
//        System.out.println(a);

//        List<String> list = Lists.newArrayList("");
//        System.out.println(String.format("【%s】",s));
//        System.out.println(list.size());
//        list = list.stream().filter(e -> "stu2".equals(e.getName())).collect(Collectors.toList());
//        System.out.println(list.size());


//        System.out.println(stu1.getName());
//        changeStudent(stu1);
//        System.out.println(stu1.getName());

//        List<String> stringList = new ArrayList<>();
//        stringList.add("stu1");
//        List<Integer> integerList = new ArrayList<>();
//        integerList.add(1);
//        integerList.add(2);
//        for(int i = 500; i > 0; i--){
//            integerList.parallelStream().forEach(e -> {
//                tttt(stringList, list);
//            });
//        }
//        List<String> stringList = new ArrayList<>();
//        stringList.add("aa");
//        stringList.add("aa");
//        stringList.add("bb");
//        stringList.add("bb");
//        stringList.add("cc");
//        List<String> stringList2 = new ArrayList<>();
//        stringList2.add("cc");
//        stringList2.add("dd");
////        stringList = stringList.stream().distinct().collect(Collectors.toList());
//        stringList.removeAll(stringList2);
//        stringList.stream().forEach(e -> System.out.println(e));

//        Set<String> set = new HashSet<>();
//        set.add("a");
//        set.add("b");
//        set.add("c");
//        List<String> list = new ArrayList<>();
//        list.add("a");
//        list.add("d");
//        set.addAll(list);
//        set.stream().forEach(e -> System.out.println(e));

//        StringBuffer url = new StringBuffer("192.168.10.126:8010/formengine/getRid");
//        Map<String, Object> queryMap = new HashMap<>();
//        queryMap.put("type1", "1");
//        queryMap.put("type2", "2");
//        url.append("?");
//        queryMap.entrySet().stream().forEach(e -> {
//            url.append(e.getKey());
//            url.append("=");
//            url.append(e.getValue());
//            url.append("&");
//        });
//        url.deleteCharAt(url.length() - 1);
//        System.out.println(url.toString());

//        String bcode = "NYDCHSTCS";
//        bcode = bcode.replaceAll("STCS$", "STZSSC");
//        System.out.println(new Date());
//        Double d1 = 0.1d;
//        Double d2 = 0.1d;
//        Double d3 = 0.1d;
//        BigDecimal b1 = new BigDecimal(d1.toString());
//        BigDecimal b2 = new BigDecimal(d2.toString());
//        BigDecimal b3 = new BigDecimal(d3.toString());
//        BigDecimal sum = b1.add(b2).add(b3);
//        System.out.println(sum);

//        List<String> list = Lists.newArrayList();
//        list.add("1");
//        String result = list.stream().map(String::valueOf).collect(Collectors.joining(","));
//        System.out.println(result);

//        BigDecimal bigDecimal1 = new BigDecimal(4.1235);
//        BigDecimal bigDecimal2 = new BigDecimal("4.1235");
//        BigDecimal bigDecimal3 = new BigDecimal("0");
//        System.out.println(new BigDecimal(String.valueOf(bigDecimal1.doubleValue())));

//        String s = "132&&updateBy";
//        System.out.println(s.substring(0, s.lastIndexOf("&&updateBy")));

//        List<String> formIds = new ArrayList<>();
//        formIds.add("1230");
//        formIds.add("1231");
//        formIds.add("1232");
//        formIds.add("1233");
//        formIds.add("1234");
//        String s = StringUtils.join(formIds,",");
//        System.out.println(s);

//        String ss = " ";
//        ss = ss.trim();
//        String s = "sfg.545";
//        Date date = new Date();
//        System.out.println(date.getTime());

//        Class clazz = JobmApproval.class;
//        Field[] allFields = clazz.getDeclaredFields();
//        //获取public参数
//        Field[] publicFields = clazz.getFields();
//        //获取private参数
//        Set<String> privateFieldName = Sets.newHashSet();
//        for(Field field : allFields){
//            privateFieldName.add(field.getName().toUpperCase());
//        }
//        for(Field field : publicFields){
//            privateFieldName.remove(field.getName().toUpperCase());
//        }
//        privateFieldName.remove("approvalDateStr".toUpperCase());
//        for(String s : privateFieldName){
//            System.out.println(s);
//        }
//
//        String s = "$RE:approval||f742cbb1-dd47-4051-b1e5-57d049597247::SCYJ_SCJG.FG_HQ||" + "";
//        String[] ss = s.split("\\|\\|");
//        System.out.println(ss.length);

//        String ssss = "NWJ5QTViK0Q1YkN4NWJteTc3eU01TGlONWJ5QTViK0Q1YkN4NXJhbQ==";
//        String s2 = "6LCD6Jaq";
//        System.out.println(new String(Base64.getDecoder().decode(s2)));

//        String str2 = "Tmt4RFJEWktZWEU9";
//        System.out.println(Base64.getEncoder().encodeToString(str2.getBytes()));

//        LinkedList<String> linkedList = new LinkedList<>();
//        linkedList.add("1");
//        linkedList.add("2");
//        linkedList.add("3");
//        linkedList.removeLast();
//        linkedList.size();

//        List<String> list173 = Lists.newArrayList();
//        String filePath173 = "D:/study/workspace/MavenDemo/src/main/java/demo/666";
//        File file173 = new File(filePath173);
//        String encoding = "utf-8";
//        try (InputStreamReader read = new InputStreamReader(new FileInputStream(file173), encoding);
//             BufferedReader bufferedReader = new BufferedReader(read)) {
//            //判断文件是否存在
//            if (file173.isFile() && file173.exists()) {
//                String lineTxt;
//                while ((lineTxt = bufferedReader.readLine()) != null) {
//                    list173.add(lineTxt);
//                }
//            } else {
//                System.out.println("找不到指定的文件");
//            }
//        } catch (Exception e) {
//            System.out.println("读取文件内容出错");
//        }
//
//        List<String> list40 = Lists.newArrayList();
//        String filePath40 = "D:/study/workspace/MavenDemo/src/main/java/demo/777";
//        File file40 = new File(filePath40);
//        try (InputStreamReader read = new InputStreamReader(new FileInputStream(file40), encoding);
//             BufferedReader bufferedReader = new BufferedReader(read)) {
//            //判断文件是否存在
//            if (file40.isFile() && file40.exists()) {
//                String lineTxt;
//                while ((lineTxt = bufferedReader.readLine()) != null) {
//                    list40.add(lineTxt);
//                }
//            } else {
//                System.out.println("找不到指定的文件");
//            }
//        } catch (Exception e) {
//            System.out.println("读取文件内容出错");
//        }
//
//        List<String> list = Lists.newArrayList();
//        list173.forEach(e -> {
//            if(!list40.contains(e)){
//                list.add(e);
//            }
//        });
//        list.size();
//        Demo demo = new Demo();
//        boolean flag = demo.isFlag();
//        System.out.println("current flag:" + flag);
//        new Thread(() -> {
//            try {
//                Thread.sleep(2000);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//            demo.setFlag(true);
//            System.out.println("Thread set flag:" + demo.isFlag());
//        }).start();
//        Something something = new Something("1", false, 1);
//        ThreadPoolExecutor threadPoolExecutor = new ThreadPoolExecutor(2, 3, 5, TimeUnit.SECONDS, new ArrayBlockingQueue<>(5));
//        final CountDownLatch countDownLatch = new CountDownLatch(1);
//        threadPoolExecutor.execute(() -> {
////            something.setS1("2");
//            try {
//                Thread.sleep(2000);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//            demo.setFlag(true);
//            System.out.println("Thread set flag:" + demo.isFlag());
//            countDownLatch.countDown();
//        });
//        threadPoolExecutor.execute(() -> {
//            something.setB1(true);
//            countDownLatch.countDown();
//        });
//        threadPoolExecutor.execute(() -> {
//            something.setI1(2);
//            countDownLatch.countDown();
//        });
//        countDownLatch.await();
//        threadPoolExecutor.shutdown();
//        if("2".equals(something.getS1())){
//            System.out.println("设置成功");
//        }
//        if(something.isB1()){
//            System.out.println("设置成功");
//        }
//        if(2 == (something.getI1())){
//            System.out.println("设置成功");
//        }
//        CompletableFuture<Void> completableFuture = CompletableFuture.supplyAsync(() -> {
//            try {
//                Thread.sleep(2000);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//            demo.setFlag(true);
//            System.out.println("Thread set flag:" + demo.isFlag());
//            return null;
//        });
//        completableFuture.join();
//
//        while (true) {
//            boolean flag2 = demo.isFlag();
//            if(flag2){
//                System.out.println("detected flag changed");
//                break;
//            }
//        }
//        threadPoolExecutor.shutdown();


//        List<String> list = new ArrayList<>();
//        list.add("1");
//        list.add("1");
//        list.add("1");
//        list.add("1");
//        list.add("1");
//        list.add("1");
//        list.add("1");
//        list.parallelStream().forEach(e -> {
//            try {
//                Thread.sleep(1000);
//                System.out.println("parallelStream里面");
//            } catch (InterruptedException interruptedException) {
//                interruptedException.printStackTrace();
//            }
//        });
//        System.out.println("parallelStream外面");

        //多张表，String为表名，List是该表的多条记录
//        Map<String, List<String>> table = Maps.newHashMap();
//        table.entrySet().parallelStream().forEach(data -> {
//            CompletableFuture[] completableFutures = data.getValue().stream().map(e -> CompletableFuture.runAsync(() -> {
//                //插入数据
//            }, ThreadPoolBuilder.getRightThreadPool())).toArray(CompletableFuture[]::new);
//            CompletableFuture.allOf(completableFutures).join();
//        });
//        ConcurrentLinkedQueue<Throwable> errorQueue = new ConcurrentLinkedQueue<>();
//        CompletableFuture<String> task2 = CompletableFuture.supplyAsync(() -> {
//            try {
//                Thread.sleep(1000);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//            System.out.println(Thread.currentThread().getName());
//            String str = null;
//            str.split(",");
//            return "studentTest";
//        }, ThreadPoolBuilder.getRightThreadPool()).exceptionally(throwable -> {
//            errorQueue.add(throwable);
//            return null;
//        }).whenComplete((s, throwable) -> {
//            ThreadPoolBuilder.getRightThreadPool().shutdown();
//        });
//        String jid = task2.get();
//        if(errorQueue.size() != 0){
//            throw errorQueue.poll();
//        }

        //测试提交

//        System.out.println((1 << 2) | (1 << 1));
//
//        AtomicInteger atomicInteger = new AtomicInteger(0);
//        int j = atomicInteger.addAndGet(1);
//        System.out.println(j);
//
//        String sss = "4";
//        int bitNum = 0;
//        for(String str : sss.split(",")){
//            bitNum = bitNum | Integer.parseInt(str);
//        }
//        System.out.println(bitNum);

//        ProcStateEnum procStateEnum = ProcStateEnum.valuesOf("在办");
//        System.out.println(procStateEnum.getValue());
//
//        String s = "OR ss";
//        System.out.println(s.substring(2));
//
//        System.out.println(Integer.toBinaryString(4));
//        System.out.println(Integer.toBinaryString(4).lastIndexOf("0"));
//        System.out.println("10000".length() - "10010".lastIndexOf("1"));
//        Integer.toBinaryString(4).lastIndexOf("1");

//        List<String> stringList = ImmutableList.of("1","2");
//        String s = "leetcode";
//        System.out.println(s.indexOf("leet"));
//        System.out.println(s.substring(8).isEmpty());

//        DateTimeFormatter dateTimeFormatter = DateTimeFormatter.ofPattern("YYYY-MM-dd");
//        Date date = new Date();
//        Instant inst = date.toInstant();
//        ZoneId zoneId = ZoneId.systemDefault();
//        ZonedDateTime zonedDateTime = inst.atZone(zoneId);
//        LocalDateTime localDateTime = zonedDateTime.toLocalDateTime();
//        System.out.println(dateTimeFormatter.format(localDateTime));

//        StringBuffer stringBuffer = new StringBuffer();
//        for (int i = 0; i < 5; i++) {
//            stringBuffer.insert(stringBuffer.length(), i);
//        }
//        System.out.println(stringBuffer.toString());

//        System.out.println(Integer.toBinaryString(0));
//        System.out.println(Integer.toBinaryString(1).length() - 1);
//        System.out.println(Integer.toBinaryString(2));
//        System.out.println(Integer.toBinaryString(4));
//        System.out.println(Integer.toBinaryString(8));
//        System.out.println(Integer.toBinaryString(16));
//        System.out.println(Integer.toBinaryString(32));

//        int[] intArr = new int[]{1,2,3};
//        Integer[] integerArr = new Integer[]{1,2,3};
//        List intList = Arrays.asList(intArr);
//        List<Integer> integerList = Arrays.asList(integerArr);
//        System.out.println(intList.get(0));
//        System.out.println(integerList.get(0));
//        System.out.println(integerList.get(1));
//        System.out.println(intList.get(1));

//        int i = 1;
//        int a;
//        a = i++;
//        System.out.println(i + ":" + a);

//        CompletableFuture<Void> task1 = CompletableFuture.supplyAsync(() -> {
//            System.out.println("task1开始");
//            try {
//                Thread.sleep(1000);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//            System.out.println("task1结束");
//            return null;
//        }, ThreadPoolBuilder.getRightThreadPool());
//        CompletableFuture<Void> task2 = CompletableFuture.supplyAsync(() -> {
//            System.out.println("task2开始");
//            String s = null;
//            s.split(",");
//            return null;
//        }, ThreadPoolBuilder.getRightThreadPool());
//        CompletableFuture<Void> future = CompletableFuture.allOf(task1, task2);
//        future.join();
//        System.out.println("主线程结束");

//        String s = "/formdesigner-web/generateForm.html?formId=83fb202e-f957-437c-8873-2071d0a7cd38&type=5";
//        int idx = s.indexOf("formId=");
//        System.out.println(idx);
//        String ss = s.substring(idx + 7, idx + 43);
//        System.out.println(ss);

//
//        Demo demo = new Demo();
//        int i = demo.getInteger();
//        System.out.println(i);

//        List<Integer> list66 = new ArrayList<>();
//        list66.add(1);
//        list66.add(2);
//        list66.add(3);
//        list66.add(4);
//        list66.add(5);
//        list66.parallelStream().forEach(e -> {
//            if (e == 3) {
//                throw new FeignException.BadRequest("", null);
//            }
//        });

//        String s = "123456";
//        System.out.println(s.getBytes());
//
//        List<String> list33 = Arrays.asList("1", null, "2", null).stream().filter(e -> !StringUtils.isEmpty(e)).collect(Collectors.toList());
//        System.out.println(list33);

//        String[] all = new String[]{"1", "2"};
//        all = Arrays.copyOf(all, 4);
//        all[2] = "3";
//        String str = StringUtils.join(all, "::");
//        System.out.println(str);

//        String s = "A man, a plan, a canal: Panama";
//        System.out.println(s.toLowerCase());

//        List<Integer> o1 = new ArrayList<>();
//        o1.add(1);
//        o1.add(2);
//        List<Integer> o2 = o1.stream().filter(e -> e == 1).collect(Collectors.toList());
//        o2.add(3);
//        o2.add(4);
//        System.out.println(o1.size() + " " + o2.size());

//        System.out.println("开始前");
//        new Thread(() -> {
//            try {
//                Thread.sleep(1000L);
//                System.out.println(Thread.currentThread() + " 正在运行中");
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//        }).start();
//        System.out.println("结束");

//        Student student = new Student();
//        System.out.println(student.getAge() == 1);
        SimpleDateFormat simpleDateFormat=new SimpleDateFormat("yyyy-MM-dd");
        Long l = 1690387200000L;
        Date date=new Date(l);
        System.out.println(simpleDateFormat.format(date));
    }
}
