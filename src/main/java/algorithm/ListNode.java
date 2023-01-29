package algorithm;

/**
 * @author 吴嘉烺
 * @description
 * @date 2022/10/9
 */
public class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int x) {
        val = x;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}
