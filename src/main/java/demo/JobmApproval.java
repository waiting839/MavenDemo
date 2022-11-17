package demo;

import java.util.Date;

/**
 * @author 吴嘉烺
 * @description
 * @date 2022/6/6
 */
public class JobmApproval {
    /**
     * 数据库表名.
     */
    public static final String TABLE_NAME = "JOBM_APPROVE";

    public static final String PROPERTY_SPLIT = "::";

    public static final String ONEJOBM_SPLIT = "||";

    public static final String ONEJOBM_SPLIT_UN = "\\|\\|";
    /**
     * 标识
     */
    private String rid;
    /**
     * 业务受理号
     */
    private String jid;
    /**
     * 签名人唯一标识
     */
    private String userId = "";
    /**
     * 签名人真实姓名
     */
    private String realName = "";
    /**
     * 签名时间
     */
    private Date approvalDate = new Date();
    /**
     * 签名时间展示值（不入库）
     */
    private String approvalDateStr = "";
    /**
     * 签名意见
     */
    private String approvalOpinion = "";
    /**
     * 是否同意
     */
    private String isAgree = "";

    /**
     * 环节id
     */
    private String actKey = "";

    /**
     * 环节名称
     */
    private String actName = "";

    /**
     * 控件名称
     */
    private String fname = "";
//
//	/**
//	 * 控件类型
//	 */
//	private String ftype = "";

    /**
     * 签名印章
     */
    private String approvalStamp = "";
//
//	private Integer sort = 1;

    /**
     * 任务id
     */
    private String taskId = "";

    /**
     * 关联id
     */
    private String relationId = "";
}
