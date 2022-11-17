package demo;

import java.util.Optional;

/**
 * @author 作者 吴嘉烺
 * @description 类描述
 * @date 创建时间 2021/9/8
 */
public class OptionalDemo {

    class ZooFlat{

        private DogFlat dogFlat = new DogFlat();

        public Optional<DogFlat> getDog(){
            return Optional.ofNullable(dogFlat);
        }
    }

    class DogFlat{

        private Integer age = 10;

        private String name = null;

        public Optional<Integer> getAge(){
            return Optional.ofNullable(age);
        }

        public String getName(){
            return Optional.ofNullable(name).orElse("defaultName");
        }
    }

    public void test(){
        ZooFlat zooFlat = new ZooFlat();
        Optional.ofNullable(zooFlat).map(z -> z.getDog()).flatMap(d -> d.get().getAge())
                .ifPresent(age -> System.out.println(age));
        Optional.ofNullable(zooFlat).map(z -> z.getDog())
                .ifPresent(d -> System.out.println(d.get().getName()));
    }

    public static void main(String[] args) {
        OptionalDemo optionalDemo = new OptionalDemo();
        optionalDemo.test();
        Integer a = null;
        System.out.println(a);
    }
}
