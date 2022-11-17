package demo;

import java.util.Map;

/**
 * @author 作者 吴嘉烺
 * @description 类描述
 * @date 创建时间 2021/9/9
 */
public class Student extends Person{

    public final static String person = "666";

    private String name;

    private Integer age;

    private Map<String, Student> studentMap;

    public Student(String name, Integer age) {
        this.name = name;
        this.age = age;
    }

    public Student() {

    }

    public Student(String name, Integer age, String name1, Integer age1, Map<String, Student> studentMap) {
        super(name, age);
        this.name = name1;
        this.age = age1;
        this.studentMap = studentMap;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public void setName(String name) {
        this.name = name;
    }

    @Override
    public Integer getAge() {
        return age;
    }

    @Override
    public void setAge(Integer age) {
        this.age = age;
    }

    @Override
    public String toString() {
        return "Student{" +
                "name='" + name + '\'' +
                ", age=" + age +
                '}';
    }

    public Map<String, Student> getStudentMap() {
        return studentMap;
    }

    public void setStudentMap(Map<String, Student> studentMap) {
        this.studentMap = studentMap;
    }

    public static void main(String[] args) {
        Student student = new Student("张三", 10);

    }
}
