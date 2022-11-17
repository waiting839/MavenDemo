package demo;

import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author 吴嘉烺
 * @description
 * @date 2022/9/14
 */
@Data
@NoArgsConstructor
public class Something {
    private String s1;
    private boolean b1;
    private Integer i1;

    public Something(String s1, boolean b1, Integer i1) {
        this.s1 = s1;
        this.b1 = b1;
        this.i1 = i1;
    }
}
