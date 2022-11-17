package demo;

/**
 * @author 作者 吴嘉烺
 * @description 类描述
 * @date 创建时间 2021/9/16
 */
public class Person {

    private String name;

    private Integer age;

    public Person(){}

    public Person(String name, Integer age) {
        this.name = name;
        this.age = age;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Integer getAge() {
        return age;
    }

    public void setAge(Integer age) {
        this.age = age;
    }

    @Override
    public String toString() {
        return "Person{" +
                "name='" + name + '\'' +
                ", age=" + age +
                '}';
    }
}
