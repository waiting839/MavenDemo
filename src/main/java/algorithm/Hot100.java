package algorithm;

import com.sun.org.apache.bcel.internal.generic.RETURN;

import java.util.*;

/**
 * @author 吴嘉烺
 * @description
 * @date 2023/3/9
 */
public class Hot100 {
    /**
     * 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
     * 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
     * 你可以按任意顺序返回答案。
     * 输入：nums = [2,7,11,15], target = 9
     * 输出：[0,1]
     * 解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1]
     * @param nums
     * @param target
     * @return
     */
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{map.get(target - nums[i]), i};
            }
            map.put(nums[i], i);
        }
        return new int[]{};
    }

    /**
     * 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
     * 请你将两个数相加，并以相同形式返回一个表示和的链表。
     * 你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
     * 输入：l1 = [2,4,3], l2 = [5,6,4]
     * 输出：[7,0,8]
     * 解释：342 + 465 = 807.
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int carry = 0;
        ListNode res = new ListNode();
        ListNode head = res;
        while (l1 != null || l2 != null) {
            int n1 = l1 == null ? 0 : l1.val;
            int n2 = l2 == null ? 0 : l2.val;
            int sum = n1 + n2 + carry;
            head.next = new ListNode(sum % 10);
            head = head.next;
            carry = sum / 10;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry > 0) {
            head.next = new ListNode(carry);
        }
        return res.next;
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
        int tmp = 0;
        int res = 0;
        for(int j = 0; j < s.length(); j++){
            int i = map.getOrDefault(s.charAt(j), -1);
            map.put(s.charAt(j), j);
            // s.charAt[j]和s.charAt[i]，如果i在上一个char的最大字串长度（tmp）内，则s.charAt[j]的最大字串长度是j - i，
            // 如果如果i在上一个char的最大字串长度（tmp）外，则则s.charAt[j]的最大字串长度是tmp + 1
            // tmp是记录当前下标的最大长度，所以j - i <= tmp的时候tmp取j - 1
            tmp = j - i > tmp ? tmp + 1 : j - i;
            res = Math.max(res, tmp);
        }
        return res;
    }

    /**
     * 给你一个字符串 s，找到 s 中最长的回文子串。
     * 如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。
     * 输入：s = "babad"
     * 输出："bab"
     * 解释："aba" 同样是符合题意的答案。
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        int n = s.length();
        int maxL = 1;
        int begin = 0;
        char[] chars = s.toCharArray();
        boolean[][] dp = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
        }
        for (int l = 2; l <= n; l++) {
            for (int i = 0; i < n; i++) {
                int j = i + l - 1;
                if (j > n) {
                    break;
                }
                if (chars[i] != chars[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                if (dp[i][j] && j - i + 1 > maxL) {
                    begin = i;
                    maxL = j - i + 1;
                }
            }
        }
        return s.substring(begin, maxL + begin);
    }

    /**
     * 给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
     * 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
     * 返回容器可以储存的最大水量。
     * 说明：你不能倾斜容器
     * 输入：[1,8,6,2,5,4,8,3,7]
     * 输出：49
     * 解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int res = 0;
        while (left < right) {
            res = Math.max(res, (left - right) * Math.min(height[left], height[right]));
            if (height[left] > height[right]) {
                right--;
            } else {
                left++;
            }
        }
        return res;
    }

    /**
     * 给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，
     * 同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。
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
        int n = nums.length;
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < n - 2; i++) {
            if (nums[i] > 0) {
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int j = i + 1;
            int k = n - 1;
            while (j < k) {
                if (-nums[i] - nums[k] == nums[j]) {
                    res.add(Arrays.asList(new Integer[]{nums[i], nums[j], nums[k]}));
                    j++;
                    k--;
                    while (j < k && nums[j - 1] == nums[j]) {
                        j++;
                    }
                    while (j < k && nums[k + 1] == nums[k]) {
                        k--;
                    }
                } else if (-nums[i] - nums[k] > nums[j]) {
                    j++;
                } else {
                    k--;
                }
            }
        }
        return res;
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
        if (head == null) {
            return null;
        }
        int size = 0;
        ListNode node = head;
        while (node != null) {
            size++;
            node = node.next;
        }
        node = head;
        for (int i = 0; i < size - n - 1; i++) {
            node = node.next;
        }
        node.next = node.next.next;
        return head;
    }

    /**
     * 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
     * 有效字符串需满足：
     * 左括号必须用相同类型的右括号闭合。
     * 左括号必须以正确的顺序闭合。
     * 每个右括号都有一个对应的相同类型的左括号。
     * 输入：s = "()"
     * 输出：true
     * @param s
     * @return
     */
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '{' || c == '[') {
                stack.push(c);
            } else {
                if (stack.isEmpty()) {
                    return false;
                }
                if ((c == ')' && stack.peek() == '(') ||
                        (c == '}' && stack.peek() == '{') ||
                        (c == ']' && stack.peek() == '[')) {
                    stack.pop();
                }
            }
        }
        return stack.isEmpty();
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
        ListNode res = new ListNode();
        ListNode head = res;
        while (list1 != null && list2 != null) {
            if (list1.val >= list2.val) {
                head.next = new ListNode(list2.val);
                list2 = list2.next;
            } else {
                head.next = new ListNode(list1.val);
                list1 = list1.next;
            }
            head = head.next;
        }
        if (list1 != null) {
            head.next = list1;
        }
        if (list2 != null) {
            head.next = list2;
        }
        return res.next;
    }

    /**
     * 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
     * 输入：n = 3
     * 输出：["((()))","(()())","(())()","()(())","()()()"]
     * @param n
     * @return
     */
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        generateParenthesis_help(n, new StringBuilder(), res, 0, 0);
        return res;
    }

    private void generateParenthesis_help(int n, StringBuilder stringBuilder, List<String> list, int left, int right) {
        if (stringBuilder.length() == n * 2) {
            list.add(stringBuilder.toString());
            return;
        }
        //穷举法，左右括号的数量用left right去计数
        if (left < n) {
            stringBuilder.append("(");
            generateParenthesis_help(n, stringBuilder, list, left + 1, right);
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);
        }
        if (left > right) {
            stringBuilder.append(")");
            generateParenthesis_help(n, stringBuilder, list, left, right + 1);
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);
        }
    }

    /**
     * 整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。
     * 例如，arr = [1,2,3] ，以下这些都可以视作 arr 的排列：[1,2,3]、[1,3,2]、[3,1,2]、[2,3,1] 。
     * 整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，
     * 那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。
     * 例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。
     * 类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。
     * 而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。
     * 给你一个整数数组 nums ，找出 nums 的下一个排列。
     * 必须 原地 修改，只允许使用额外常数空间
     * 输入：nums = [1,2,3]
     * 输出：[1,3,2]
     * @param nums
     */
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        //从后面找出降序前的一个数字
        while (i >= 0 && nums[i] >= nums[i - 1]) {
            i--;
        }
        if (i >= 0) {
            int j = nums.length - 1;
            //从后面找出第一个比nums[i]大的数字，然后交换
            while (j >= 0 && nums[i] >= nums[j]) {
                j--;
            }
            swap(nums, i, j);
        }
        //再反转，相当于升序
        reverse(nums, i + 1);
    }

    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    private void reverse(int[] nums, int start) {
        int i = start;
        int j = nums.length - 1;
        while (i < j) {
            int tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
            i++;
            j--;
        }
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
        int i = 0;
        int j = nums.length - 1;
        while (i < j) {
            int mid = i + (j - i) / 2;
            if (nums[mid] == target) {
                return mid;
            //判断mid前面是不是有序的，有序即是后面才发生旋转
            } else if (nums[mid] >= nums[i]) {
                //前面有序，target小于nums[mid]而且target大于等于nums[i]才能确保一定是j减少
                //后面无序的直接i增加
                if (nums[mid] > target && target >= nums[i]) {
                    j = mid - 1;
                } else {
                    i = mid + 1;
                }
            } else {
                //后面无序，target大于nums[mid]而且target小于等于nums[j]才能确保一定是i增加
                if (nums[mid] < target && target <= nums[j]) {
                    i = mid + 1;
                } else {
                    j = mid - 1;
                }
            }
        }
        return -1;
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
        int i = 0, j = nums.length - 1;
        //找出target在nums中第一个位置（左边界）
        while (i <= j) {
            int mid = i + (j - i) / 2;
            if (nums[mid] >= target) {
                j = mid - 1;
            } else {
                i = mid + 1;
            }
        }
        if (i >= nums.length || nums[i] != target) {
            return new int[]{-1, -1};
        } else {
            int[] res = new int[2];
            res[0] = i;
            while (i + 1 < nums.length && nums[i + 1] == target) {
                i++;
            }
            res[1] = i;
            return res;
        }
    }

    /**
     * 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，
     * 并以列表形式返回。你可以按 任意顺序 返回这些组合。
     * candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。
     * 对于给定的输入，保证和为 target 的不同组合数少于 150 个
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
        combinationSum_help(candidates, target, 0, new LinkedList<>(), 0);
        return combinationSum_res;
    }

    private void combinationSum_help(int[] candidates, int target, int sum, LinkedList<Integer> path, int start) {
        if (sum == target) {
            combinationSum_res.add(new ArrayList<>(path));
            return;
        }
        //通过i = start 去保证不会回头选择前面的数
        for (int i = start; i < candidates.length && sum + candidates[i] <= target; i++) {
            sum += candidates[i];
            path.add(candidates[i]);
            combinationSum_help(candidates, target, sum, path, i);
            sum -= candidates[i];
            path.removeLast();
        }
    }

    /**
     * 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
     * @param nums
     * @return
     */
    List<List<Integer>> permute_res = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        permute_help(nums, new LinkedList<>());
        return permute_res;
    }

    private void permute_help(int[] nums, LinkedList<Integer> path) {
        if (path.size() == nums.length) {
            permute_res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (path.contains(nums[i])) {
                continue;
            }
            path.add(nums[i]);
            permute_help(nums, path);
            path.removeLast();
        }
    }

    /**
     * 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
     * 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
     * 输入：matrix =
     * [[1,2,3],
     * [4,5,6],
     * [7,8,9]]
     * 输出：
     * [[7,4,1],
     * [8,5,2],
     * [9,6,3]]
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        //先上下翻转
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n - i - 1][j];
                matrix[n - i - 1][j] = tmp;
            }
        }
        //再沿左对角线翻转
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }
    }

    /**
     * 给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
     * 字母异位词 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。
     * 输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
     * 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
     * @param strs
     * @return
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> res = new ArrayList<>();
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            //用排序好的异位词作为key
            String key = new String(chars);
            if (map.containsKey(key)) {
                map.get(key).add(str);
            } else {
                List<String> list = new ArrayList<>();
                list.add(str);
                map.put(key, list);
            }
        }
        map.entrySet().forEach(e -> {
            res.add(e.getValue());
        });
        return res;
    }

    /**
     * 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
     * 子数组 是数组中的一个连续部分。
     * 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
     * 输出：6
     * 解释：连续子数组 [4,-1,2,1] 的和最大，为 6
     * @param nums
     * @return
     */
    public int maxSubArray(int[] nums) {
        int res = nums[0];
        int tmp = nums[0];
        for (int i = 1; i < nums.length; i++) {
            tmp = Math.max(nums[i], nums[i] + tmp);
            res = Math.max(res, tmp);
        }
        return res;
    }

    /**
     * 给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。
     * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
     * 判断你是否能够到达最后一个下标。
     * 输入：nums = [2,3,1,1,4]
     * 输出：true
     * 解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
     * @param nums
     * @return
     */
    public boolean canJump(int[] nums) {
        int maxRight = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i <= maxRight) {
                //计算可达最长长度
                maxRight = Math.max(maxRight, i + nums[i]);
                if (maxRight >= nums.length - 1) {
                    return true;
                }
            }
        }
        return false;
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
        if (intervals == null || intervals[0] == null) {
            return new int[][]{};
        }
        List<int[]> list = new ArrayList<>();
        //左区间排序升序
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        for (int i = 0; i < intervals.length; i++) {
            int left = intervals[i][0];
            int right = intervals[i][1];
            if (list.isEmpty() || list.get(list.size() - 1)[1] < left) {
                list.add(new int[]{left, right});
            } else {
                list.get(list.size() - 1)[1] = Math.max(list.get(list.size() - 1)[1], right);
            }
        }
        return list.toArray(new int[][]{});
    }

    /**
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
     * 问总共有多少条不同的路径？
     * 输入：m = 3, n = 7
     * 输出：28
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = 1;
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    /**
     * 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
     * 说明：每次只能向下或者向右移动一步。
     * @param grid
     * @return
     */
    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = grid[i][j];
                } else if (i == 0) {
                    dp[i][j] = grid[i][j] + dp[i][j - 1];
                } else if (j == 0) {
                    dp[i][j] = grid[i][j] + dp[i - 1][j];
                } else {
                    dp[i][j] = grid[i][j] + Math.min(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    /**
     * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
     * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }
        int cur = 2;
        int pre = 1;
        while (n > 2) {
            int tmp = cur;
            cur = cur + pre;
            pre = tmp;
            n--;
        }
        return cur;
    }

    /**
     * 给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
     * 我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
     * 必须在不使用库内置的 sort 函数的情况下解决这个问题。
     * 输入：nums = [2,0,2,1,1,0]
     * 输出：[0,0,1,1,2,2]
     * @param nums
     */
    public void sortColors(int[] nums) {
        sortColors_help(nums, 0, nums.length - 1);
    }

    private void sortColors_help(int[] nums, int i, int j) {
        if (i >= j) {
            return;
        }
        int index = sortColors_quickSort(nums, i, j);
        sortColors_quickSort(nums, i, index - 1);
        sortColors_quickSort(nums, index + 1, j);
    }

    private int sortColors_quickSort(int[] nums, int i, int j) {
        int k = nums[i];
        int l = i + 1;
        int r = j;
        while (l < r) {
            while (l < r && nums[r] > k) {
                r--;
            }
            while (l < r && nums[l] <= k) {
                l++;
            }
            if (r < l) {
                swap(nums, l, r);
            }
        }
        nums[i] = nums[r];
        nums[r] = k;
        return r;
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
        subsets_help(nums, new LinkedList<>(), 0);
        return subsets_res;
    }

    private void subsets_help(int[] nums, LinkedList<Integer> path, int start) {
        subsets_res.add(new ArrayList<>(path));
        if(start == nums.length){
            return;
        }
        for (int i = start; i < nums.length; i++) {
            path.add(nums[i]);
            subsets_help(nums, path, i + 1);
            path.removeLast();
        }
    }

    /**
     * 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
     * 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
     * 输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
     * 输出：true
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (exist_help(board, i, j, word, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean exist_help(char[][] board, int i, int j, String word, int index) {
        if (i < 0 || i > board.length - 1 || j < 0 || j > board[0].length - 1 || board[i][j] != word.charAt(index)) {
            return false;
        }
        if (index == word.length() - 1) {
            return true;
        }
        char tmp = board[i][j];
        board[i][j] = '/';
        boolean res =  exist_help(board, i + 1, j, word, index + 1)
                || exist_help(board, i - 1, j, word, index + 1)
                || exist_help(board, i, j + 1, word, index + 1)
                || exist_help(board, i, j - 1, word, index + 1);
        board[i][j] = tmp;
        return res;
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
     * 给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。
     * 输入：n = 3
     * 输出：5
     * @param n
     * @return
     */
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                //除去根节点一共有i - 1个节点
                //假如右边从0个开始增加到i，左边就有i - j个，右边就有j - 1个
                int left = dp[i - j];
                int right = dp[j - 1];
                //再组合相乘即可
                dp[i] += left * right;
            }
        }
        return dp[n];
    }

    /**
     * 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
     * 有效 二叉搜索树定义如下：
     * 节点的左子树只包含 小于 当前节点的数。
     * 节点的右子树只包含 大于 当前节点的数。
     * 所有左子树和右子树自身必须也是二叉搜索树。
     * @param root
     * @return
     */
    List<Integer> isValidBST_res = new ArrayList<>();
    public boolean isValidBST(TreeNode root) {
        //通过中序遍历获取所有值，二叉搜索树的中序遍历是升序排序的
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
        //判断是否同时为空和对应的值是否相等
        if ((left == null || right == null) || (left.val != right.val)) {
            return false;
        }
        return isSymmetric_help(left.left, right.right) && isSymmetric_help(left.right, right.left);
    }

    /**
     * 给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）
     * 输入：root = [3,9,20,null,null,15,7]
     * 输出：[[3],[9,20],[15,7]]
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        Queue<TreeNode> queue = new ArrayDeque<>();
        List<List<Integer>> res = new ArrayList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            List<Integer> tmp = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                tmp.add(node.val);
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
            }
            res.add(tmp);
        }
        return res;
    }

    /**
     * 给定一个二叉树，找出其最大深度。
     * 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
     * 说明: 叶子节点是指没有子节点的节点。
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
     * 给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。
     * 输入: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
     * 输出: [3,9,20,null,null,15,7]
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        Map<Integer, Integer> inorderMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++){
            inorderMap.put(inorder[i], i);
        }
        return buildTree_help(preorder, inorderMap, 0, 0, inorder.length - 1);
    }

    private TreeNode buildTree_help(int[] preorder, Map<Integer, Integer> inorderMap, int root, int left, int right) {
        if (left > right) {
            return null;
        }
        TreeNode node = new TreeNode(preorder[root]);
        //i为inorder中根节点的下标
        int i = inorderMap.get(preorder[root]);
        //root是相对于predoder，找根节点
        //left和right分别是inorder的左右边界
        //root + 1是指左子树的下一个根节点（相对于preorder）
        node.left = buildTree_help(preorder, inorderMap, root + 1, left, i - 1);
        //root + 1 + i - left是指右子树的下一个根节点，i - left是指左子树有这么多个节点
        node.right = buildTree_help(preorder, inorderMap, root + 1 + i - left, i + 1, right);
        return node;
    }

    /**
     * 给你二叉树的根结点 root ，请你将它展开为一个单链表：
     * 展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
     * 展开后的单链表应该与二叉树 先序遍历 顺序相同。
     * @param root
     */
    public void flatten(TreeNode root) {
        TreeNode cur = root;
        //寻找前驱节点，把cur的right放到左子树的第一个最后一个right节点
        while (cur != null) {
            if (cur.left != null) {
                TreeNode next = cur.left;
                TreeNode predecessor = next;
                while (predecessor.right != null) {
                    predecessor = predecessor.right;
                }
                predecessor.right = cur.right;
                cur.left = null;
                cur.right = next;
            }
            cur = cur.right;
        }
    }

    /**
     * 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
     * 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
     * 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
     * 输入：[7,1,5,3,6,4]
     * 输出：5
     * 解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     *      注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int res = 0;
        //记录最低价格
        int minPrice = prices[0];
        for (int price : prices) {
            minPrice = Math.min(minPrice, price);
            res = Math.max(res, price - minPrice);
        }
        return res;
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
        int res = 0;
        for (int num : nums) {
            set.add(num);
        }
        for (int num : nums) {
            //如果是num - 1则会计算所有情况，如果是num + 1则会计算值最大的那个
            if (!set.contains(num + 1)) {
                int curNum = num;
                int curLength = 1;
                while (set.contains(curNum - 1)) {
                    curNum--;
                    curLength++;
                }
                res = Math.max(res, curLength);
            }
        }
        return res;
    }

    /**
     * 给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
     * 你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。
     * 输入：nums = [2,2,1]
     * 输出：1
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        int res = 0;
        for (int num : nums) {
            res ^= num;
        }
        return res;
    }

    /**
     * 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
     * 注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
     * 输入: s = "leetcode", wordDict = ["leet", "code"]
     * 输出: true
     * 解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                //截取字符串去判断
                if (dp[j] && set.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    /**
     * 给你一个链表的头节点 head ，判断链表中是否有环。
     * 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，
     * 评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。
     * 如果链表中存在环 ，则返回 true 。 否则，返回 false
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
        ListNode slow = head;
        ListNode fast = head.next;
        while (slow != fast) {
            //快慢指针，如果遍历到null则为没有环
            if (fast == null || fast.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }

    /**
     * 给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
     * 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。
     * 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。
     * 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
     * 不允许修改 链表。
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) {
            return null;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (true) {
            if (fast == null || fast.next == null) {
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                break;
            }
        }
        fast = head;
        while (fast != slow) {
            slow = slow.next;
            fast = fast.next;
        }
        return fast;
    }

    /**
     * 给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。
     * @param head
     * @return
     */
    public ListNode sortList(ListNode head) {
        sortList(head, null);
        return null;
    }

    private ListNode sortList(ListNode head, ListNode tail) {
        if (head == null) {
            return null;
        }
        if (head.next == tail) {
            head.next = null;
            return head;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast != tail) {
            slow = slow.next;
            fast = fast.next;
            if (fast != tail) {
                fast = fast.next;
            }
        }
        ListNode mid = slow;
        ListNode head1 = sortList(head, mid);
        ListNode head2 = sortList(mid, tail);
        ListNode sort = mergeList(head1, head2);
        return sort;
    }

    private ListNode mergeList(ListNode head1, ListNode head2) {
        ListNode node = new ListNode();
        ListNode head = node;
        while (head1 != null && head2 != null) {
            if (head1.val > head2.val) {
                head.next = new ListNode(head2.val);
                head2 = head2.next;
            } else {
                head.next = new ListNode(head1.val);
                head1 = head1.next;
            }
            head = head.next;
        }
        if (head1 != null) {
            head.next = head1;
        }
        if (head2 != null) {
            head.next = head2;
        }
        return node.next;
    }

    /**
     * 给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
     * 测试用例的答案是一个 32-位 整数。
     * 子数组 是数组的连续子序列
     * 输入: nums = [2,3,-2,4]
     * 输出: 6
     * 解释: 子数组 [2,3] 有最大乘积 6。
     * @param nums
     * @return
     */
    public int maxProduct(int[] nums) {
        int res = nums[0];
        int max = nums[0];
        int min = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int maxTmp = max;
            int minTmp = min;
            //nums[i]可能是正的也可能是负的，所以要先计算maxTmp * nums[i], minTmp * nums[i]中最大的，再和自己做比较
            max = Math.max(nums[i], Math.max(maxTmp * nums[i], minTmp * nums[i]));
            min = Math.min(nums[i], Math.min(maxTmp * nums[i], minTmp * nums[i]));
            res = Math.max(res, max);
        }
        return res;
    }

    /**
     * 给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。
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
     * 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。
     * 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        int candidate = 0;
        int sum = 0;
        for (int num : nums) {
            if (sum == 0) {
                candidate = num;
            }
            sum += candidate == num ? 1 : -1;
        }
        return candidate;
    }

    /**
     * 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
     * 如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
     * 输入：[1,2,3,1]
     * 输出：4
     * 解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     *      偷窃到的最高金额 = 1 + 3 = 4 。
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (i - 1 > 0) {
                dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
            } else {
                dp[i] = Math.max(dp[i - 1], nums[i]);
            }
        }
        return dp[nums.length - 1];
    }

    /**
     * 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
     * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
     * 此外，你可以假设该网格的四条边均被水包围。
     * 输入：grid = [
     *   ["1","1","0","0","0"],
     *   ["1","1","0","0","0"],
     *   ["0","0","1","0","0"],
     *   ["0","0","0","1","1"]
     * ]
     * 输出：3
     * @param grid
     * @return
     */
    public int numIslands(char[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    res++;
                    numIslands_help(grid, i, j);
                }
            }
        }
        return res;
    }

    private void numIslands_help(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j > grid[0].length || grid[i][j] == '0') {
            return;
        }
        grid[i][j] = '0';
        numIslands_help(grid, i + 1, j);
        numIslands_help(grid, i - 1, j);
        numIslands_help(grid, i, j + 1);
        numIslands_help(grid, i, j - 1);
    }

    /**
     * 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
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
            ListNode tmp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = tmp;
        }
        return pre;
    }

    /**
     * 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
     * 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
     * 你必须设计并实现时间复杂度为 O(n) 的算法解决此问题
     * 输入: [3,2,1,5,6,4], k = 2
     * 输出: 5
     * @param nums
     * @param k
     * @return
     */
    public int findKthLargest(int[] nums, int k) {
        int target = nums.length - k;
        int l = 0;
        int r = nums.length - 1;
        while (true) {
            int index = findKthLargest_help(nums, l ,r);
            if (index == target) {
                return nums[index];
            } else if (index > target) {
                r = index - 1;
            } else if (index < target) {
                l = index + 1;
            }
        }
    }

    private int findKthLargest_help(int[] nums, int l, int r) {
        int k = nums[l];
        int i = l;
        int j = r;
        while (i < j) {
            while (i < j && nums[j] >= k) {
                j--;
            }
            while (i < j && nums[i] <= k) {
                i++;
            }
            if (i < j) {
                swap(nums, i , j);
            }
        }
        nums[l] = nums[j];
        nums[j] = k;
        return j;
    }

    /**
     * 在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。
     * 输入：matrix =
     * [["1","0","1","0","0"],
     * ["1","0","1","1","1"],
     * ["1","1","1","1","1"],
     * ["1","0","0","1","0"]]
     * 输出：4
     * @param matrix
     * @return
     */
    public int maximalSquare(char[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] dp = new int[m][n];
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '0') {
                    continue;
                } else {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = 1 + Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                    }
                }
                res = Math.max(res, dp[i][j]);
            }
        }
        return res * res;
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
        TreeNode right = root.right;
        root.left = invertTree(right);
        root.right = invertTree(left);
        return root;
    }

    /**
     * 给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false
     * 输入：head = [1,2,2,1]
     * 输出：true
     * @param head
     * @return
     */
    public boolean isPalindrome(ListNode head) {
        Stack<Integer> stack = new Stack<>();
        ListNode node = head;
        while (node != null) {
            stack.push(node.val);
            node = node.next;
        }
        node = head;
        while (node != null) {
            if (node.val != stack.pop()) {
                return false;
            }
        }
        return true;
        //可以使用快慢指针然后返回后半部分链表达到空间复杂度O(1)
    }

    /**
     * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
     * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，
     * 最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }
        //往下寻找节点p和q，如果p和q在不同子树，则left和right都不为空，如果在同一个子树，则另一则会为空
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null) {
            return right;
        }
        if (right == null) {
            return left;
        }
        return root;
    }

    /**
     * 给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
     * 题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。
     * 请不要使用除法，且在 O(n) 时间复杂度内完成此题。
     * 输入: nums = [1,2,3,4]
     * 输出: [24,12,8,6]
     * @param nums
     * @return
     */
    public int[] productExceptSelf(int[] nums) {
        int[] res = new int[nums.length];
        int p = 1;
        int q = 1;
        //从左往右相乘，紧乘到自己下标前的
        for (int i = 0; i < nums.length; i++) {
            res[i] = p;
            p *= nums[i];
        }
        //从右往左相乘，紧乘到自己下标后的
        for (int i = nums.length - 1; i >= 0; i--) {
            res[i] *= q;
            q *= nums[i];
        }
        return res;
    }

    /**
     * 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
     * 每行的元素从左到右升序排列。
     * 每列的元素从上到下升序排列。
     * 输入：matrix =
     * [[1,4,7,11,15],
     * [2,5,8,12,19],
     * [3,6,9,16,22],
     * [10,13,14,17,24],
     * [18,21,23,26,30]], target = 5
     * 输出：true
     * @param matrix
     * @param target
     * @return
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length - 1;
        int n = 0;
        while (m >= 0 && n < matrix[0].length) {
            if (target == matrix[m][n]) {
                return true;
            } else if (target > matrix[m][n]) {
                n++;
            } else if (target < matrix[m][n]) {
                m--;
            }
        }
        return false;
    }

    /**
     * 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
     * 完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
     * 输入：n = 12
     * 输出：3
     * 解释：12 = 4 + 4 + 4
     * @param n
     * @return
     */
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            int min = Integer.MAX_VALUE;
            //找j*j前一个最小数量的，找到后+1即可
            for (int j = 1; j * j <= i; j++) {
                min = Math.min(min, dp[i - j * j]);
            }
            dp[i] = min + 1;
        }
        return dp[n];
    }

    /**
     * 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
     * 请注意 ，必须在不复制数组的情况下原地对数组进行操作。
     * 输入: nums = [0,1,0,3,12]
     * 输出: [1,3,12,0,0]
     * @param nums
     */
    public void moveZeroes(int[] nums) {
        //p指向下标为0的，
        //q去找下标不为0的
        int p = 0;
        int q = 0;
        while (q < nums.length) {
            if (nums[q] != 0) {
                swap(nums, p, q);
                p++;
            }
            q++;
        }
    }

    /**
     * 给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。
     * 假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。
     * 你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。
     * 输入：nums = [1,3,4,2,2]
     * 输出：2
     * @param nums
     * @return
     */
    public int findDuplicate(int[] nums) {
        //快慢指针，类似环形链表找到循环节点
        int slow = nums[0];
        int fast = nums[0];
        while (true) {
            slow = nums[slow];
            fast = nums[nums[fast]];
            if (slow == fast) {
                break;
            }
        }
        fast = nums[0];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    /**
     * 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
     * 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
     * 输入：nums = [10,9,2,5,3,7,101,18]
     * 输出：4
     * 解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
     * @param nums
     * @return
     */
    public int lengthOfLIS(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length];
        int res = 1;
        dp[0] = 1;
        for (int i = 1; i < nums.length; i++) {
            dp[i] = 1;
            //两重遍历，如果nums[i] > nums[j]则取dp[i]和dp[j] + 1 的最大值
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    /**
     * 给定一个整数数组prices，其中第  prices[i] 表示第 i 天的股票价格 。
     * 设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
     * 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
     * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     * 输入: prices = [1,2,3,0,2]
     * 输出: 3
     * 解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
     * @param prices
     * @return
     */
    public int maxProfit2(int[] prices) {
        // 代表持有股票
        int p1 = -prices[0];
        // 代表昨天卖出股票，在冷却期
        int p2 = 0;
        // 代表昨天之前卖出股票，不在冷却期
        int p3 = 0;
        for (int i = 1; i < prices.length; i++) {
            int newp1 = Math.max(p1, p3 - prices[i]);
            int newp2 = prices[i] + p1;
            int newp3 = Math.max(p2, p3);
            p1 = newp1;
            p2 = newp2;
            p3 = newp3;
        }
        return Math.max(p2, p3);
    }

    /**
     * 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
     * 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
     * 你可以认为每种硬币的数量是无限的。
     * 输入：coins = [1, 2, 5], amount = 11
     * 输出：3
     * 解释：11 = 5 + 5 + 1
     * @param coins
     * @param amount
     * @return
     */
    public int coinChange(int[] coins, int amount) {
        coinChange_help(coins, amount, new int[amount]);
        return 0;
    }

    private int coinChange_help(int[] coins, int amount, int[] count) {
        if (amount == 0) {
            return 0;
        }
        if (amount < 0) {
            return -1;
        }
        if (count[amount - 1] != 0) {
            return count[amount - 1];
        }
        int res = Integer.MAX_VALUE;
        for (int coin : coins) {
           int sub = coinChange_help(coins, amount - coin, count);
           if (sub == -1) {
               continue;
           }
           res = Math.min(res, sub + 1);
        }
        count[amount - 1] = res == Integer.MAX_VALUE ? -1 : res;
        return count[amount - 1];
    }

    /**
     * 小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 root 。
     * 除了 root 之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。
     * 如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。
     * 给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额
     * @param root
     * @return
     */
    public int rob(TreeNode root) {
        int[] res = rob_help(root);
        return Math.max(res[0], res[1]);
    }

    private int[] rob_help(TreeNode root) {
        if (root == null) {
            return new int[]{0, 0};
        }
        //数组第一个表示选择了，第二个表示不选择
        int[] left = rob_help(root.left);
        int[] right = rob_help(root.right);
        //选择当前节点则加上左右子树不选的值
        int select = root.val + left[1] + right[1];
        //不选择当前节点则取左右子树中选和不选中最大值
        int notSelect = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        return new int[]{select, notSelect};
    }

    /**
     * 给你一个整数 n ，对于 0 <= i <= n 中的每个 i ，计算其二进制表示中 1 的个数 ，返回一个长度为 n + 1 的数组 ans 作为答案。
     * 输入：n = 2
     * 输出：[0,1,1]
     * 解释：
     * 0 --> 0
     * 1 --> 1
     * 2 --> 10
     * @param n
     * @return
     */
    public int[] countBits(int n) {
        int[] res = new int[n + 1];
        res[0] = 0;
        int highBit = 0;
        for (int i = 1; i <= n; i++) {
            //判断是否高位，例如3（11）的高位是2（10）
            if ((i & (i - 1)) == 0) {
                highBit = i;
            }
            //当前值就是高位前面的所有加上高位
            res[i] = res[i - highBit] + 1;
        }
        return res;
    }

    /**
     * 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
     * 输入: nums = [1,1,1,2,2,3], k = 2
     * 输出: [1,2]
     * @param nums
     * @param k
     * @return
     */
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        //使用最小堆
        PriorityQueue<int[]> priorityQueue = new PriorityQueue<>(Comparator.comparingInt(o -> o[1]));
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (priorityQueue.size() < k) {
                priorityQueue.add(new int[]{entry.getKey(), entry.getValue()});
            } else {
                if (priorityQueue.peek()[1] < entry.getValue()) {
                    priorityQueue.poll();
                    priorityQueue.add(new int[]{entry.getKey(), entry.getValue()});
                }
            }
        }
        int[] res = new int[k];
        for (int i = 0; i < k; i++) {
            res[i] = priorityQueue.poll()[0];
        }
        return res;
    }

    /**
     * 给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
     * 输入：nums = [1,5,11,5]
     * 输出：true
     * 解释：数组可以分割成 [1, 5, 5] 和 [11] 。
     * @param nums
     * @return
     */
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        //少于两个数返回false
        if (n < 2) {
            return false;
        }
        int sum = 0;
        int maxNum = 0;
        for (int num : nums) {
            sum += num;
            maxNum = Math.max(maxNum, num);
        }
        //总和是奇数，返回false
        if (sum % 2 == 1) {
            return false;
        }
        int target = sum / 2;
        //最大的那个数大于总和的一半，返回false
        if (maxNum > target) {
            return false;
        }
        boolean[][] dp = new boolean[n][target + 1];
        //初始化target为0的情况，即不选任何数的情况下都是true
        for (int i = 0; i < n; i++) {
            dp[i][0] = true;
        }
        //初始化第一个选的数，target为nums[0]时选他刚好等于，即为true
        dp[0][nums[0]] = true;
        //完全背包
        for (int i = 1; i < n; i++) {
            for (int j = 1; j <= target; j++) {
                if (j >= nums[i]) {
                    //j >= nums[i]即是可以选择nums[i]
                    dp[i][j] = dp[i - 1][j] | dp[i][j - nums[i]];
                } else {
                    //不能选择的情况只能根据i - 1的情况顺延
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[n - 1][target];
    }

    /**
     * 给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
     * 路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
     * @param root
     * @param targetSum
     * @return
     */
    public int pathSum(TreeNode root, int targetSum) {
        Map<Long, Integer> prefix = new HashMap<>();
        //自己一个节点刚好也是一条路径
        prefix.put(0L, 1);
        return pathSum_help(root, targetSum, 0L, prefix);
    }

    private int pathSum_help(TreeNode node, int targetSum, long cur, Map<Long, Integer> prefix) {
        if (node == null) {
            return 0;
        }
        cur += node.val;
        //前缀和 - target如果有则说明刚好相等
        int res = prefix.getOrDefault(cur - targetSum, 0);
        //存储前缀和
        prefix.put(cur, prefix.getOrDefault(cur, 0) + 1);
        res += pathSum_help(node.left, targetSum, cur, prefix);
        res += pathSum_help(node.right, targetSum, cur, prefix);
        //回溯
        prefix.put(cur, prefix.get(cur) - 1);
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
        int sLen = s.length();
        int pLen = p.length();
        if (sLen < pLen) {
            return new ArrayList<>();
        }
        List<Integer> res = new ArrayList<>();
        //使用滑动窗口
        int[] sArr = new int[26];
        int[] pArr = new int[26];
        //先将p长度的字母写进两个数组，然后比较
        for (int i = 0; i < pLen; i++) {
            sArr[s.charAt(i) - 'a']++;
            pArr[s.charAt(i) - 'a']++;
        }
        if (Arrays.equals(sArr, pArr)) {
            res.add(0);
        }
        //从0开始删除sArr的值，向右滑动，一个一个判断
        for (int i = 0; i < sLen - pLen; i++) {
            sArr[s.charAt(i) - 'a']--;
            sArr[s.charAt(i + pLen) - 'a']++;
            if (Arrays.equals(sArr, pArr)) {
                res.add(i + 1);
            }
        }
        return res;
    }

    /**
     * 给你一个含 n 个整数的数组 nums ，其中 nums[i] 在区间 [1, n] 内。请你找出所有在 [1, n] 范围内但没有出现在 nums 中的数字，
     * 并以数组的形式返回结果。
     * 输入：nums = [4,3,2,7,8,2,3,1]
     * 输出：[5,6]
     * @param nums
     * @return
     */
    public List<Integer> findDisappearedNumbers(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n + 1];
        List<Integer> res = new ArrayList<>();
        for (int num : nums) {
            dp[num]++;
        }
        for (int i = 1; i <= n; i++) {
            if (dp[i] == 0) {
                res.add(i);
            }
        }
        return res;
    }

    /**
     * 两个整数之间的 汉明距离 指的是这两个数字对应二进制位不同的位置的数目。
     * 给你两个整数 x 和 y，计算并返回它们之间的汉明距离。
     * @param x
     * @param y
     * @return
     */
    public int hammingDistance(int x, int y) {
        String str = String.valueOf(x ^ y);
        int res = 0;
        for (char c : str.toCharArray()) {
            if (c == '1') {
                res++;
            }
        }
        return res;
    }

    /**
     * 给你一个整数数组 nums 和一个整数 target 。
     * 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：
     * 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
     * 返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。
     * 输入：nums = [1,1,1,1,1], target = 3
     * 输出：5
     * 解释：一共有 5 种方法让最终目标和为 3 。
     * -1 + 1 + 1 + 1 + 1 = 3
     * +1 - 1 + 1 + 1 + 1 = 3
     * +1 + 1 - 1 + 1 + 1 = 3
     * +1 + 1 + 1 - 1 + 1 = 3
     * +1 + 1 + 1 + 1 - 1 = 3
     * @param nums
     * @param target
     * @return
     */
    int findTargetSumWays_res = 0;
    public int findTargetSumWays(int[] nums, int target) {
        findTargetSumWays_help(nums, target, 0, 0);
        return 0;
    }

    private void findTargetSumWays_help(int[] nums, int target, int sum, int index) {
        if (index == nums.length) {
            if (sum == target) {
                findTargetSumWays_res++;
                return;
            }
        } else {
            findTargetSumWays_help(nums, target, sum + nums[index], index + 1);
            findTargetSumWays_help(nums, target, sum - nums[index], index + 1);
        }
    }

    /**
     * 给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），
     * 使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。
     * 提醒一下，二叉搜索树满足下列约束条件：
     * 节点的左子树仅包含键 小于 节点键的节点。
     * 节点的右子树仅包含键 大于 节点键的节点。
     * 左右子树也必须是二叉搜索树。
     * @param root
     * @return
     */
    int convertBST_sum = 0;
    public TreeNode convertBST(TreeNode root) {
        if (root == null) {
            return null;
        }
        convertBST(root.right);
        convertBST_sum += root.val;
        root.val = convertBST_sum;
        convertBST(root.left);
        return root;
    }

    /**
     * 给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。
     * @param root
     * @return
     */
    int diameterOfBinaryTree_res = 0;
    public int diameterOfBinaryTree(TreeNode root) {
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
     * 给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的连续子数组的个数 。
     * 输入：nums = [1,1,1], k = 2
     * 输出：2
     * @param nums
     * @param k
     * @return
     */
    public int subarraySum(int[] nums, int k) {
        int count = 0;
        int pre = 0;
        //存取前缀和，key为前缀和的值，value是次数
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int i = 0; i < nums.length; i++) {
            pre += nums[i];
            if (map.containsKey(pre - k)) {
                //例如[2,0,0,2]，当遍历到第四位时，map中有键值对{2,3}，说明可以有3对和为2的组合
                count += map.get(pre - k);
            }
            map.put(pre, map.getOrDefault(pre, 0) + 1);
        }
        return count;
    }


    public static void main(String[] args) {
        Hot100 hot100 = new Hot100();
        System.out.println(hot100.subarraySum(new int[]{2,0,0}, 2));
    }
}
