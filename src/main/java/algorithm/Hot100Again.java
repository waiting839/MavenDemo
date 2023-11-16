package algorithm;

import java.util.*;

/**
 * @author 吴嘉烺
 * @description
 * @date 2023/10/9
 */
public class Hot100Again {

    /**
     * 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
     * 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
     * 你可以按任意顺序返回答案。
     * 输入：nums = [2,7,11,15], target = 9
     * 输出：[0,1]
     * 解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1]
     * @param nums
     * @param target
     * @retur
     */
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{i, map.get(target - nums[i])};
            }
            map.put(nums[i], i);
        }
        return new int[2];
    }

    /**
     * 给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
     * 字母异位词 是由重新排列源单词的所有字母得到的一个新单词。
     * 输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
     * 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
     * @param strs
     * @return
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            List<String> list = map.computeIfAbsent(new String(chars), k -> new ArrayList<>());
            list.add(str);
            map.put(new String(chars), list);
        }
        List<List<String>> result = new ArrayList<>();
        map.forEach((k, v) -> result.add(v));
        return result;
    }

    /**
     * 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
     * 请你设计并实现时间复杂度为 O(n) 的算法解决此问题。
     * 输入：nums = [100,4,200,1,3,2]
     * 输出：4
     * 解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
     * @param nums
     * @return
     */
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int res = 0;
        for (int num : nums) {
            if (set.contains(num + 1)) {
                continue;
            }
            int tmp = num;
            int sum = 1;
            while (set.contains(tmp - 1)) {
                tmp--;
                sum++;
            }
            res = Math.max(res, sum);
        }
        return res;
    }

    /**
     * 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
     * 请注意 ，必须在不复制数组的情况下原地对数组进行操作。
     * 输入: nums = [0,1,0,3,12]
     * 输出: [1,3,12,0,0]
     * @param nums
     */
    public void moveZeroes(int[] nums) {
        int j = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                swap(nums, i, j);
                j++;
            }
        }
    }

    /**
     * 给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
     * 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
     * 返回容器可以储存的最大水量。
     * 说明：你不能倾斜容器。
     * 输入：[1,8,6,2,5,4,8,3,7]
     * 输出：49
     * 解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        int res = 0;
        int i = 0;
        int j = height.length - 1;
        while (i < j) {
            int area = (j - i) * Math.min(height[i], height[j]);
            res = Math.max(res, area);
            if (height[i] < height[j]) {
                i++;
            } else {
                j--;
            }
        }
        return res;
    }

    /**
     * 给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，
     * 同时还满足 nums[i] + nums[j] + nums[k] == 0 。请
     * 你返回所有和为 0 且不重复的三元组。
     * 注意：答案中不可以包含重复的三元组。
     * 输入：nums = [-1,0,1,2,-1,-4]
     * 输出：[[-1,-1,2],[-1,0,1]]
     * 解释：
     * nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
     * nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
     * nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
     * 不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
     * 注意，输出的顺序和三元组的顺序并不重要。
     * @param nums
     * @return
     */
    public List<List<Integer>> threeSum(int[] nums) {
        if (nums == null || nums.length < 3) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < nums.length - 2; i++) {
            if (nums[i] > 0) {
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int j = nums.length - 1;
            int k = i + 1;
            int target = -nums[i];
            while (k < j) {
                if (target - nums[j] == nums[k]) {
                    List<Integer> list = Arrays.asList(nums[i], nums[k++], nums[j--]);
                    res.add(list);
                    while (k < j && nums[k] == nums[k - 1]) {
                        k++;
                    }
                    while (k < j && nums[j] == nums[j + 1]) {
                        j--;
                    }
                } else if (target - nums[j] < nums[k]) {
                    j--;
                } else if (target - nums[j] > nums[k]){
                    k++;
                }
            }
        }
        return res;
    }

    /**
     * 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
     * 输入: s = "abcabcbb"
     * 输出: 3
     * 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int res = 0;
        int tmp = 0;
        for (int i = 0; i < s.length(); i++) {
            int index = map.getOrDefault(s.charAt(i), -1);
            tmp = i - index > tmp ? tmp + 1 : i - index;
            res = Math.max(res, tmp);
            map.put(s.charAt(i), i);
        }
        return res;
    }

    /**
     * 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
     * 异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。
     * 输入: s = "cbaebabacd", p = "abc"
     * 输出: [0,6]
     * 解释:
     * 起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
     * 起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
     * @param s
     * @param p
     * @return
     */
    public List<Integer> findAnagrams(String s, String p) {
        if (s.length() < p.length()) {
            return new ArrayList<>();
        }
        int sLen = s.length();
        int pLen = p.length();
        int[] sArr = new int[26];
        int[] pArr = new int[26];
        for (int i = 0; i < pLen; i++) {
            sArr[s.charAt(i) - 'a']++;
            pArr[p.charAt(i) - 'a']++;
        }
        List<Integer> res = new ArrayList<>();
        if (Arrays.equals(sArr, pArr)) {
            res.add(0);
        }
        for (int i = 1; i <= sLen - pLen; i++) {
            sArr[s.charAt(i - 1) - 'a']--;
            sArr[s.charAt(i + pLen - 1) - 'a']++;
            if (Arrays.equals(sArr, pArr)) {
                res.add(i);
            }
        }
        return res;
    }

    /**
     * 给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的连续子数组的个数 。
     * 子数组是数组中元素的连续非空序列。
     * 输入：nums = [1,1,1], k = 2
     * 输出：2
     * @param nums
     * @param k
     * @return
     */
    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int pre = 0;
        int res = 0;
        for (int num : nums) {
            pre += num;
            if (map.containsKey(pre - k)) {
                res += map.get(pre - k);
            }
            map.put(pre, map.getOrDefault(pre, 0) + 1);
        }
        return res;
    }

    /**
     * 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
     * 子数组 是数组中的一个连续部分。
     * 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
     * 输出：6
     * 解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
     * @param nums
     * @return
     */
    public int maxSubArray(int[] nums) {
        int res = nums[0];
        int cur = nums[0];
        for (int i = 1; i < nums.length; i++) {
            cur = Math.max(cur + nums[i], nums[i]);
            res = Math.max(res, cur);
        }
        return res;
    }

    /**
     * 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。
     * 请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
     * 输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
     * 输出：[[1,6],[8,10],[15,18]]
     * 解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
     * @param intervals
     * @return
     */
    public int[][] merge(int[][] intervals) {
        if (intervals == null || intervals.length < 1) {
            return new int[0][0];
        }
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        LinkedList<int[]> list = new LinkedList<>();
        for (int[] arr : intervals) {
            if (list.size() == 0) {
                list.add(arr);
            } else {
                int[] pre = list.getLast();
                if (pre[1] < arr[0]) {
                    list.add(arr);
                } else {
                    pre[1] = Math.max(pre[1], arr[1]);
                }
            }
        }
        return list.toArray(new int[][]{});
    }

    /**
     * 给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
     * 输入: nums = [1,2,3,4,5,6,7], k = 3
     * 输出: [5,6,7,1,2,3,4]
     * 解释:
     * 向右轮转 1 步: [7,1,2,3,4,5,6]
     * 向右轮转 2 步: [6,7,1,2,3,4,5]
     * 向右轮转 3 步: [5,6,7,1,2,3,4]
     * @param nums
     * @param k
     */
    public void rotate(int[] nums, int k) {
        int length = nums.length;
        int[] arr = new int[length];
        for (int i = 0; i < length; i++) {
            arr[(i + k) % length] = nums[i];
        }
        System.arraycopy(arr, 0, nums, 0, length);
    }

    /**
     * 给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
     * 题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。
     * 请 不要使用除法，且在 O(n) 时间复杂度内完成此题
     * 输入: nums = [1,2,3,4]
     * 输出: [24,12,8,6]
     * @param nums
     * @return
     */
    public int[] productExceptSelf(int[] nums) {
        if (nums == null) {
            return new int[0];
        }
        int p = 1;
        int q = 1;
        int n = nums.length;
        int[] res = new int[n];
        for (int i = 0; i < n; i++) {
            res[i] = p;
            p *= nums[i];
        }
        for (int i = n - 1; i >= 0; i--) {
            res[i] *= q;
            q *= nums[i];
        }
        return res;
    }

    /**
     * 给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
     * 输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
     * 输出：[[1,0,1],[0,0,0],[1,0,1]]
     * @param matrix
     */
    public void setZeroes(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        boolean[] rowArr = new boolean[m];
        boolean[] colArr = new boolean[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    rowArr[i] = true;
                    colArr[j] = true;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            if (rowArr[i]) {
                for (int j = 0; j < n; j++) {
                    matrix[i][j] = 0;
                }
            }
        }
        for (int j = 0; j < n; j++) {
            if (colArr[j]) {
                for (int i = 0; i < m; i++) {
                    matrix[i][j] = 0;
                }
            }
        }
    }

    /**
     * 给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
     * 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
     * 输出：[1,2,3,6,9,8,7,4,5]
     * @param matrix
     * @return
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        int top = 0;
        int down = matrix.length - 1;
        int left = 0;
        int right = matrix[0].length - 1;
        List<Integer> res = new ArrayList<>();
        while (true) {
            for (int i = left; i <= right; i++) {
                res.add(matrix[top][i]);
            }
            top++;
            if (top > down) {
                break;
            }
            for (int i = top; i <= down; i++) {
                res.add(matrix[i][right]);
            }
            right--;
            if (left > right) {
                break;
            }
            for (int i = right; i >= left; i--) {
                res.add(matrix[down][i]);
            }
            down--;
            if (top > down) {
                break;
            }
            for (int i = down; i >= top; i--) {
                res.add(matrix[i][left]);
            }
            left++;
            if (left > right) {
                break;
            }
        }
        return res;
    }

    /**
     * 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
     * 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < n; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n - i - 1][j];
                matrix[n - i - 1][j] = tmp;
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }
    }

    /**
     * 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。
     * 输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
     * 输出：Intersected at '8'
     * 解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。
     * 从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,6,1,8,4,5]。
     * 在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
     * — 请注意相交节点的值不为 1，因为在链表 A 和链表 B 之中值为 1 的节点 (A 中第二个节点和 B 中第三个节点) 是不同的节点。
     * 换句话说，它们在内存中指向两个不同的位置，而链表 A 和链表 B 中值为 8 的节点 (A 中第三个节点，B 中第四个节点) 在内存中指向相同的位置。
     * @param headA
     * @param headB
     * @return
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode A = headA;
        ListNode B = headB;
        while (A != B) {
            A = A == null ? headB : A.next;
            B = B == null ? headA : B.next;
        }
        return A;
    }

    /**
     * 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
     * 输入：head = [1,2,3,4,5]
     * 输出：[5,4,3,2,1]
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    /**
     * 给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。
     * 输入：head = [1,2,2,1]
     * 输出：true
     * @param head
     * @return
     */
    public boolean isPalindrome(ListNode head) {
        if (head == null) {
            return true;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null) {
            if (fast.next != null) {
                fast = fast.next.next;
                slow = slow.next;
            } else {
                break;
            }
        }
        ListNode cur = slow;
        ListNode pre = null;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        ListNode node = head;
        while (node != null && pre != null) {
            if (node.val != pre.val) {
                return false;
            }
            node = node.next;
            pre = pre.next;
        }
        return true;
    }

    /**
     * 给你一个链表的头节点 head ，判断链表中是否有环。
     * 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。
     * 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。
     * 仅仅是为了标识链表的实际情况。
     * 如果链表中存在环 ，则返回 true 。 否则，返回 false 。
     * 输入：head = [3,2,0,-4], pos = 1
     * 输出：true
     * 解释：链表中有一个环，其尾部连接到第二个节点。
     * @param head
     * @return
     */
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (true) {
            if (fast == null || fast.next == null) {
                return false;
            }
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                break;
            }
        }
        return true;
    }

    /**
     * 给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
     * 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，
     * 评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。
     * 注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
     * 不允许修改 链表。
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (true) {
            if (fast == null || fast.next == null) {
                return null;
            }
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                break;
            }
        }
        slow = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    /**
     * 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
     * 输入：l1 = [1,2,4], l2 = [1,3,4]
     * 输出：[1,1,2,3,4,4]
     * @param list1
     * @param list2
     * @return
     */
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode head = new ListNode();
        ListNode node = head;
        while (list1 != null && list2 != null) {
            if (list1.val > list2.val) {
                node.next = new ListNode(list2.val);
                list2 = list2.next;
            } else {
                node.next = new ListNode(list1.val);
                list1 = list1.next;
            }
            node = node.next;
        }
        if (list1 != null) {
            node.next = list1;
        }
        if (list2 != null) {
            node.next = list2;
        }
        return head.next;
    }

    /**
     * 给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
     * 输入：head = [1,2,3,4,5], n = 2
     * 输出：[1,2,3,5]
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if(head == null){
            return null;
        }
        int length = 0;
        ListNode node = head;
        while (node != null) {
            node = node.next;
            length++;
        }
        if (length == n) {
            return head.next;
        }
        node = head;
        for (int i = 0; i < length - n - 1; i++) {
            node = node.next;
        }
        node.next = node.next.next;
        return head;
    }

    /**
     * 给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。
     * 你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
     * 输入：head = [1,2,3,4]
     * 输出：[2,1,4,3]
     * @param head
     * @return
     */
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode node = new ListNode();
        node.next = head;
        ListNode pre = node;
        ListNode cur = head;
        while (cur != null && cur.next != null) {
            ListNode next = cur.next.next;
            cur.next.next = cur;
            pre.next = cur.next;
            cur.next = next;
            pre = cur;
            cur = cur.next;
        }
        return node.next;
    }

    /**
     * 给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。
     * 输入：root = [1,null,2,3]
     * 输出：[1,3,2]
     * @param root
     * @return
     */
    List<Integer> inorderTraversal_res = new ArrayList<>();
    public List<Integer> inorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        inorderTraversal(root.left);
        inorderTraversal_res.add(root.val);
        inorderTraversal(root.right);
        return inorderTraversal_res;
    }

    /**
     * 给定一个二叉树 root ，返回其最大深度。
     * 二叉树的 最大深度 是指从根节点到最远叶子节点的最长路径上的节点数。
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    /**
     * 给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。
     * @param root
     * @return
     */
    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode left = root.left;
        root.left = root.right;
        root.right = left;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }

    /**
     * 给你一个二叉树的根节点 root ， 检查它是否轴对称。
     * 输入：root = [1,2,2,3,4,4,3]
     * 输出：true
     * @param root
     * @return
     */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isSymmetric_help(root.left, root.right);
    }

    private boolean isSymmetric_help(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if ((left == null || right == null) || left.val != right.val) {
            return false;
        }
        return isSymmetric_help(left.right, right.left) && isSymmetric_help(left.left, right.right);
    }

    /**
     * 给你一棵二叉树的根节点，返回该树的 直径 。
     * 二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。
     * 两节点之间路径的 长度 由它们之间边数表示。
     * 输入：root = [1,2,3,4,5]
     * 输出：3
     * 解释：3 ，取路径 [4,2,1,3] 或 [5,2,1,3] 的长度。
     * @param root
     * @return
     */
    int diameterOfBinaryTree_res = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        if (root == null) {
            return 0;
        }
        diameterOfBinaryTree_help(root);
        return diameterOfBinaryTree_res;
    }

    private int diameterOfBinaryTree_help(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = diameterOfBinaryTree_help(root.left);
        int right = diameterOfBinaryTree_help(root.right);
        diameterOfBinaryTree_res = Math.max(diameterOfBinaryTree_res, left + right);
        return Math.max(left, right) + 1;
    }

    /**
     * 给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int n = queue.size();
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                TreeNode node = queue.poll();
                list.add(node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            res.add(list);
        }
        return res;
    }

    /**
     * 给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。
     * 高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
     * @param nums
     * @return
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBST_help(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBST_help(int[] nums, int left, int right) {
        if (left > right) {
            return null;
        }
        int mid = left + (right - left) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = sortedArrayToBST_help(nums, left, mid - 1);
        node.right = sortedArrayToBST_help(nums, mid + 1, right);
        return node;
    }

    /**
     * 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
     * 有效 二叉搜索树定义如下：
     * 节点的左子树只包含 小于 当前节点的数。
     * 节点的右子树只包含 大于 当前节点的数。
     * 所有左子树和右子树自身必须也是二叉搜索树。
     * 输入：root = [2,1,3]
     * 输出：true
     * @param root
     * @return
     */
    List<Integer> isValidBST_res = new ArrayList<>();
    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        isValidBST_help(root);
        for (int i = 1; i < isValidBST_res.size(); i++) {
            if (isValidBST_res.get(i) <= isValidBST_res.get(i - 1)) {
                return false;
            }
        }
        return true;
    }

    private void isValidBST_help(TreeNode root) {
        if (root == null) {
            return;
        }
        isValidBST_help(root.left);
        isValidBST_res.add(root.val);
        isValidBST_help(root.right);
    }

    /**
     * 给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）
     * @param root
     * @param k
     * @return
     */
    List<Integer> kthSmallest_res = new ArrayList<>();
    public int kthSmallest(TreeNode root, int k) {
        kthSmallest_help(root);
        return kthSmallest_res.get(k - 1);
    }

    private void kthSmallest_help(TreeNode root) {
        if (root == null) {
            return;
        }
        kthSmallest_help(root.left);
        kthSmallest_res.add(root.val);
        kthSmallest_help(root.right);
    }

    /**
     * 给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
     * @param root
     * @return
     */
    public List<Integer> rightSideView(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        Queue<TreeNode> queue = new LinkedList<>();
        List<Integer> res = new ArrayList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int n = queue.size();
            for (int i = 0; i < n; i++) {
                TreeNode node = queue.poll();
                if (i == 0) {
                    res.add(node.val);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
                if (node.left != null) {
                    queue.offer(node.left);
                }
            }
        }
        return res;
    }

    /**
     * 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
     * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
     * 此外，你可以假设该网格的四条边均被水包围。
     * @param grid
     * @return
     */
    public int numIslands(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (numIslands_help(grid, i, j)) {
                    res++;
                }
            }
        }
        return res;
    }

    private boolean numIslands_help(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != '1') {
            return false;
        }
        grid[i][j] = '0';
        numIslands_help(grid, i + 1, j);
        numIslands_help(grid, i - 1, j);
        numIslands_help(grid, i, j + 1);
        numIslands_help(grid, i, j - 1);
        return true;
    }

    /**
     * 在给定的 m x n 网格 grid 中，每个单元格可以有以下三个值之一：
     * 值 0 代表空单元格；
     * 值 1 代表新鲜橘子；
     * 值 2 代表腐烂的橘子。
     * 每分钟，腐烂的橘子 周围 4 个方向上相邻 的新鲜橘子都会腐烂。
     * 返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1 。
     * 输入：grid = [[2,1,1],[1,1,0],[0,1,1]]
     * 输出：4
     * @param grid
     * @return
     */
    public int orangesRotting(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        int[][] move = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 2) {
                    queue.offer(new int[]{i, j});
                }
            }
        }
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] arr = queue.poll();
                for (int j = 0; j < 4; j++) {
                    int x = arr[0] + move[j][0];
                    int y = arr[1] + move[j][1];
                    if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == 1) {
                        grid[x][y] = 2;
                        queue.offer(new int[]{x, y});
                    }
                }
            }
            res++;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    return -1;
                }
            }
        }
        return res - 1 < 0 ? 0 : res - 1;
    }



    /**
     * 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
     * @param nums
     * @return
     */
    List<List<Integer>> permute_res = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        permute_help(nums, new ArrayList<>());
        return permute_res;
    }

    private void permute_help(int[] nums, List<Integer> list) {
        if (nums.length == list.size()) {
            permute_res.add(new ArrayList<>(list));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (list.contains(nums[i])) {
                continue;
            }
            list.add(nums[i]);
            permute_help(nums, list);
            list.remove(list.size() - 1);
        }
    }

    /**
     * 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
     * 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
     * 输入：nums = [1,2,3]
     * 输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
     * @param nums
     * @return
     */
    List<List<Integer>> subsets_res = new ArrayList<>();
    public List<List<Integer>> subsets(int[] nums) {
        subsets_help(nums, new ArrayList<>(), 0);
        return subsets_res;
    }

    private void subsets_help(int[] nums, List<Integer> list, int start) {
        subsets_res.add(new ArrayList<>(list));
        for (int i = start; i < nums.length; i++) {
            list.add(nums[i]);
            subsets_help(nums, list, i + 1);
            list.remove(list.size() - 1);
        }
    }

    /**
     * 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，
     * 并以列表形式返回。你可以按 任意顺序 返回这些组合。
     * candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。
     * 对于给定的输入，保证和为 target 的不同组合数少于 150 个。
     * 输入：candidates = [2,3,6,7], target = 7
     * 输出：[[2,2,3],[7]]
     * 解释：
     * 2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
     * 7 也是一个候选， 7 = 7 。
     * 仅有这两种组合。
     * @param candidates
     * @param target
     * @return
     */
    List<List<Integer>> combinationSum_res = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        combinationSum_help(candidates, new ArrayList<>(), 0, target, 0);
        return combinationSum_res;
    }

    private void combinationSum_help(int[] candidates, List<Integer> list, int sum, int target, int start) {
        if (sum == target) {
            combinationSum_res.add(new ArrayList<>(list));
            return;
        }
        for (int i = start; i < candidates.length; i++) {
            if (sum + candidates[i] > target) {
                continue;
            }
            list.add(candidates[i]);
            combinationSum_help(candidates, list, sum + candidates[i], target, i);
            list.remove(list.size() - 1);
        }
    }

    /**
     * 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
     * 输入：n = 3
     * 输出：["((()))","(()())","(())()","()(())","()()()"]
     * @param n
     * @return
     */
    List<String> generateParenthesis_res = new ArrayList<>();
    public List<String> generateParenthesis(int n) {
        generateParenthesis_help(n, 0, 0, new StringBuilder());
        return generateParenthesis_res;
    }

    private void generateParenthesis_help(int n, int left, int right, StringBuilder stringBuilder) {
        if (right == n) {
            generateParenthesis_res.add(stringBuilder.toString());
            return;
        }
        if (left < n) {
            stringBuilder.append("(");
            generateParenthesis_help(n, left + 1, right, stringBuilder);
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);
        }
        if (right < left) {
            stringBuilder.append(")");
            generateParenthesis_help(n, left, right + 1, stringBuilder);
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);
        }
    }

    /**
     * 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
     * 请必须使用时间复杂度为 O(log n) 的算法。
     * 输入: nums = [1,3,5,6], target = 5
     * 输出: 2
     * @param nums
     * @param target
     * @return
     */
    public int searchInsert(int[] nums, int target) {
        int l = 0;
        int r = nums.length - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return l;
    }

    /**
     * 给你一个满足下述两条属性的 m x n 整数矩阵：
     * 每行中的整数从左到右按非严格递增顺序排列。
     * 每行的第一个整数大于前一行的最后一个整数。
     * 给你一个整数 target ，如果 target 在矩阵中，返回 true ；否则，返回 false 。
     * 输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
     * 输出：true
     * @param matrix
     * @param target
     * @return
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int i = 0;
        int j = matrix[0].length;
        while (i < matrix.length && j >= 0) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] > target) {
                j--;
            } else {
                i++;
            }
        }
        return false;
    }

    /**
     * 给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
     * 如果数组中不存在目标值 target，返回 [-1, -1]。
     * 你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。
     * 输入：nums = [5,7,7,8,8,10], target = 8
     * 输出：[3,4]
     * @param nums
     * @param target
     * @return
     */
    public int[] searchRange(int[] nums, int target) {
        int l = 0;
        int r = nums.length - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                int i = mid;
                int j = mid;
                while (i > 0 && nums[i - 1] == target) {
                    i--;
                }
                while (j < nums.length - 1 && nums[j + 1] == target) {
                    j++;
                }
                return new int[]{i, j};
            }else if (nums[mid] > target) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return new int[]{-1, -1};
    }

    /**
     * 整数数组 nums 按升序排列，数组中的值 互不相同 。
     * 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，
     * 使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。
     * 例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
     * 给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
     * 你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
     * 输入：nums = [4,5,6,7,0,1,2], target = 0
     * 输出：4
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
        int l = 0;
        int r = nums.length - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[mid] >= nums[l]) {
                if (nums[mid] > target && target >= nums[l]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[r]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return -1;
    }

    /**
     * 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
     * 若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
     * 若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
     * 注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
     * 给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
     * 你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
     * 输入：nums = [3,4,5,1,2]
     * 输出：1
     * 解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
     * @param nums
     * @return
     */
    public int findMin(int[] nums) {
        int l = 0;
        int r = nums.length - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (nums[l] <= nums[mid]) {
                if (nums[l] > nums[r]) {
                    l = mid + 1;
                } else {
                    return nums[l];
                }
            }else {
                if (nums[mid] > nums[mid - 1]) {
                    r = mid - 1;
                } else {
                    return nums[mid];
                }
            }
        }
        return -1;
    }

    public static void main(String[] args) {
        Hot100Again hot100Again = new Hot100Again();
        hot100Again.searchRange(new int[]{5,7,7,8,8,10}, 9);
    }

    private void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
