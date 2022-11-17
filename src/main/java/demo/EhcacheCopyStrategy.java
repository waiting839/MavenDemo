package demo;

import net.sf.ehcache.Element;
import net.sf.ehcache.store.compound.ReadWriteCopyStrategy;

import java.io.*;

/**
 * @author 作者 吴嘉烺
 * @description 类描述
 * @date 创建时间 2021/10/8
 */
public class EhcacheCopyStrategy implements ReadWriteCopyStrategy<Element> {
    @Override
    public Element copyForWrite(Element value, ClassLoader classLoader) {
        if(value != null){
            Object temp = value.getObjectValue();
            try {
                return new Element(value.getObjectKey(),deepCopy(temp));
            } catch (IOException e) {
                e.printStackTrace();
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            }
        }
        return value;
    }

    @Override
    public Element copyForRead(Element storedValue, ClassLoader classLoader) {
        if(storedValue != null){
            Object temp = storedValue.getObjectValue();
            try {
                return new Element(storedValue.getObjectKey(),deepCopy(temp));
            } catch (ClassNotFoundException | IOException e) {
                e.printStackTrace();
            }
        }
        return storedValue;
    }

    private  Object deepCopy(Object src) throws IOException, ClassNotFoundException {
        ByteArrayOutputStream byteOut = new ByteArrayOutputStream();
        ObjectOutputStream out = new ObjectOutputStream(byteOut);
        out.writeObject(src);

        ByteArrayInputStream byteIn = new ByteArrayInputStream(byteOut.toByteArray());
        ObjectInputStream in = new ObjectInputStream(byteIn);
        return in.readObject();
    }

}
