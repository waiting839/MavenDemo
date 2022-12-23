package demo;

import java.util.Arrays;

/**
 * @author 吴嘉烺
 * @description 办理状态
 * @date 2022/12/22
 */
public enum ProcStateEnum {
    /**
     * 在办
     */
    inProcess("0", "在办"),
    /**
     * 办结
     */
    completed("1", "办结");

    private final String code;

    private final String value;

    ProcStateEnum(String code, String value) {
        this.code = code;
        this.value = value;
    }

    public String getCode() {
        return code;
    }

    public String getValue() {
        return value;
    }

    public static ProcStateEnum codeOf(String code) {
        return Arrays.stream(ProcStateEnum.values()).filter(i -> i.getCode().equals(code)).findFirst().orElse(null);
    }

    public static ProcStateEnum valuesOf(String value) {
        return Arrays.stream(ProcStateEnum.values()).filter(i -> i.getValue().equals(value)).findFirst().orElse(null);
    }
}
