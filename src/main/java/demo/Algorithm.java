package demo;

import com.google.common.collect.Lists;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author 作者 吴嘉烺
 * @description 类描述
 * @date 创建时间 2021/12/22
 */
public class Algorithm {

    public static int[] twoSum(int[] nums, int target){
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length; i++){
            if(map.containsKey((target - nums[i]))){
                return new int[]{map.get(target - nums[i]), i};
            }
            map.put(nums[i], i);
        }
        return new int[0];
    }

    public int removeElement(int[] nums, int val) {
        if(nums.length <= 0){
            return 0;
        }
        int i = 0;
        int j = nums.length - 1;
        while(i < j){
            if(nums[i] == val){
                nums[i] = nums[j];
                nums[j] = val;
                j--;
            }else{
                i++;
            }
        }
        return nums[i] == val ? i : i + 1;
    }

    /**
     * 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
     * 请必须使用时间复杂度为 O(log n) 的算法。
     * @param nums
     * @param target
     * @return
     */
    public int searchInsert(int[] nums, int target) {
        int i = 0;
        int j = nums.length - 1;
        while(i <= j){
            int k = (j - i) / 2 + i;
            if(target < nums[k]){
                j = k - 1;
            }else if(target > nums [k]){
                i = k + 1;
            }else{
                return k;
            }
        }
        return i;
    }

    /**
     * 给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。
     * 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
     * 你可以假设除了整数 0 之外，这个整数不会以零开头。
     * @param digits
     * @return
     */
    public int[] plusOne(int[] digits) {
        boolean plus = true;
        for(int i = digits.length - 1; i >= 0; i--){
            if(!plus){
                break;
            }
            if(digits[i] + 1 < 10){
                digits[i] = digits[i] + 1;
                plus = false;
            }else{
                digits[i] = 0;
            }
        }
        if(plus){
            int[] res = new int[digits.length + 1];
            res[0] = 1;
            return res;
        }
        return digits;
    }

    /**
     * 给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。
     * 高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
     * @param nums
     * @return
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        int i = 0;
        int j = nums.length - 1;
        TreeNode node = bst( nums, i ,j);
        return node;
    }

    private TreeNode bst(int[] nums, int l, int r){
        TreeNode treeNode = new TreeNode();
        if(l > r){
            return null;
        }
        int mid = (r - l) / 2 + l;
        treeNode.val = nums[mid];
        treeNode.left = bst(nums, l, mid - 1);
        treeNode.right = bst( nums, mid + 1, r);
        return treeNode;
    }

    /**
     * 给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。
     * 在「杨辉三角」中，每个数是它左上方和右上方的数的和。
     * @param numRows
     * @return
     */
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        for(int i = 0; i < numRows; i++){
            List<Integer> tmp = new ArrayList<>();
            for(int j = 0; j <= i; j++){
                if(j == 0 || j == i){
                    tmp.add(1);
                }else{
                    tmp.add(res.get(i - 1).get(j - 1) + res.get(i - 1).get(j));
                }
            }
            res.add(tmp);
        }
        return res;
    }

    /**
     * 给定一个数组 prices ，它的第i个元素prices[i]表示一支给定股票第i天的价格。
     * 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
     * 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int[] dp = new int[prices.length];
        dp[0] = prices[0];
        int max = 0;
        for(int i = 1; i < prices.length; i++){
            dp[i] = Math.min(dp[i - 1], prices[i]);
            max = Math.max(prices[i] - dp[i], max);
        }
        return max;
    }

    /**
     * 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
     * 说明：
     * 你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        int res = 0;
        for(int i = 0; i < nums.length; i++){
            res = res ^ nums[i];
        }
        return res;
    }

    /**
     * 给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数 大于⌊ n/2 ⌋的元素。
     * 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        //投票法
        int res = 0;
        int sum = 0;
        for(int num : nums){
            if(sum == 0){
                res = num;
            }
            sum += res == num ? 1 : -1;
        }
        return res;
    }

    /**
     * 给你一个整数数组 nums 。如果任一值在数组中出现 至少两次 ，返回 true ；如果数组中每个元素互不相同，返回 false 。
     * @param nums
     * @return
     */
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for(int num : nums){
            if(!set.add(num)){
                return true;
            }
        }
        return false;
    }

    /**
     * 给你一个整数数组nums 和一个整数k ，判断数组中是否存在两个 不同的索引i和j ，满足 nums[i] == nums[j] 且 abs(i - j) <= k 。如果存在，返回 true ；否则，返回 false 。
     * @param nums
     * @param k
     * @return
     */
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length; i++){
            if(map.containsKey(nums[i]) && (i - map.get(nums[i])) <= k){
                return true;
            }else{
                map.put(nums[i], i);
            }
        }
        return false;
    }

    /**
     * 给定一个 无重复元素 的有序 整数数组 nums 。
     *
     * 返回 恰好覆盖数组中所有数字 的 最小有序 区间范围列表。也就是说，nums 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个范围但不属于 nums 的数字 x 。
     * @param nums
     * @return
     */
    public List<String> summaryRanges(int[] nums) {
        List<String> res = new ArrayList<>();
        if(nums.length == 0){
            return res;
        }
        if(nums.length == 1){
            res.add(String.valueOf(nums[0]));
            return res;
        }
        int index = nums[0];
        for(int i = 1; i < nums.length; i++){
            if(nums[i - 1] + 1 != nums[i]){
                String str = (index == nums[i - 1]) ? String.valueOf(index) : index + "->" + nums[i - 1];
                res.add(str);
                index = nums[i];
            }
            if(i - 1 == nums.length){
                String str = (index == nums[i]) ? String.valueOf(index) : index + "->" + nums[i];
                res.add(str);
            }
        }
        return res;
    }

    /**
     * 给定一个包含 [0, n] 中 n 个数的数组 nums ，找出 [0, n] 这个范围内没有出现在数组中的那个数。
     * @param nums
     * @return
     */
    public int missingNumber(int[] nums) {
        if(nums.length == 0){
            return 0;
        }
        int[] tmp = new int[nums.length + 1];
        for(int num : nums){
            tmp[num] = 1;
        }
        for(int i = 0; i < tmp.length; i++){
            if(tmp[i] == 0){
                return i;
            }
        }
        return 0;
    }

    /**
     * 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
     * 请注意 ，必须在不复制数组的情况下原地对数组进行操作。
     * @param nums
     */
    public void moveZeroes(int[] nums) {
        int l = 0, r = 0;
        while(r < nums.length){
            if(nums[r] != 0){
                swap(nums, l, r);
                l++;
            }
            r++;
        }
    }

    private void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    /**
     * 给定一个长度为 n 的整数数组height。有n条垂线，第 i 条线的两个端点是(i, 0)和(i, height[i])。
     * 找出其中的两条线，使得它们与x轴共同构成的容器可以容纳最多的水。
     * 返回容器可以储存的最大水量。
     * 说明：你不能倾斜容器。
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        int res = 0;
        int l = 0, r = height.length - 1;
        while(l < r){
            res = Math.max(res, (r - l) * Math.min(height[r], height[l]));
            if(height[l] <= height[r]){
                l++;
            }else{
                r--;
            }
        }
        return res;
    }

    /**
     * 给你一个包含 n 个整数的数组nums，判断nums中是否存在三个元素 a，b，c ，使得a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
     * 注意：答案中不可以包含重复的三元组。
     * @param nums
     * @return
     */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if(nums == null || nums.length < 3){
            return res;
        }
        Arrays.sort(nums);
        for(int i = 0; i < nums.length - 2; i++){
            if(nums[i] > 0){
                break;
            }
            if(i > 0 && nums[i] == nums[i - 1]){
                continue;
            }
            int l = i + 1, r = nums.length - 1;
            //防止溢出
            int target = -nums[i];
            while(l < r){
                if(nums[l] + nums[r] == target){
                    res.add(new ArrayList<>(Arrays.asList(nums[i], nums[l], nums[r])));
                    l++;
                    r--;
                    while(l < r && nums[l] == nums[l - 1]){
                        l++;
                    }
                    while(l < r && nums[r] == nums[r + 1]){
                        r--;
                    }
                }else if(nums[l] + nums[r] < target){
                    l++;
                }else{
                    r--;
                }
            }
        }
        return res;
    }

    public Integer threeNumClosest(int[] nums, int target){
        Arrays.sort(nums);
        for(int i = 0; i < nums.length; i++){
            int l = i + 1;
            int r = nums.length - 1;
            while(l < r){

            }
        }
        return 0;
    }

    /**
     * 给定两个数组 nums1 和 nums2 ，返回 它们的交集 。输出结果中的每个元素一定是 唯一 的。我们可以 不考虑输出结果的顺序 。
     * @param nums1
     * @param nums2
     * @return
     */
    public int[] intersection(int[] nums1, int[] nums2) {
        if(nums1.length == 0 || nums2.length == 0){
            return new int[0];
        }
        Set<Integer> set1 = new HashSet<>();
        for(int num : nums1){
            set1.add(num);
        }
        Set<Integer> set2 = new HashSet<>();
        for(int num : nums2){
            set2.add(num);
        }
        Set<Integer> resSet = new HashSet<>();
        for(int num : set2){
            if(!set1.add(num)){
                resSet.add(num);
            }
        }
        int[] res = new int[resSet.size()];
        Object[] tmp = resSet.toArray();
        for(int i = 0; i < resSet.size(); i++){
            res[i] = (int)tmp[i];
        }
        return res;
    }

    public int[] intersection2(int[] nums1, int[] nums2) {
        if(nums1.length == 0 || nums2.length == 0){
            return new int[0];
        }
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int[] res = new int[nums1.length + nums2.length];
        int i = 0, j = 0, index = 0;
        while(i < nums1.length && j < nums2.length){
            if(nums1[i] == nums2[j]){
                res[index++] = nums1[i++];
                j++;
            }else if(nums1[i] < nums2[j]){
                i++;
            }else{
                j++;
            }
        }
        return Arrays.copyOfRange(res, 0, index);
    }

    /**
     * 给你一个非空数组，返回此数组中 第三大的数 。如果不存在，则返回数组中最大的数。
     * @param nums
     * @return
     */
    public int thirdMax(int[] nums) {
        TreeSet<Integer> treeSet = new TreeSet<>();
        for(int num : nums){
            treeSet.add(num);
            if(treeSet.size() > 3){
                treeSet.pollFirst();
            }
        }
        return treeSet.size() == 3 ? treeSet.first() : treeSet.last();
    }

    /**
     * 给你一个含 n 个整数的数组 nums ，其中 nums[i] 在区间 [1, n] 内。请你找出所有在 [1, n] 范围内但没有出现在 nums 中的数字，并以数组的形式返回结果。
     * @param nums
     */
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if(nums.length <= 0){
            return res;
        }
        int[] numbers = new int[nums.length + 1];
        for(int i : nums){
            numbers[i] = numbers[i] + 1;
        }
        for(int i = 1; i < numbers.length; i++){
            if(numbers[i] == 0){
                res.add(i);
            }
        }
        return res;
    }

    /**
     * 给你一个长度为 n 的整数数组，每次操作将会使 n - 1 个元素增加 1 。返回让数组所有元素相等的最小操作次数。
     * @param nums
     * @return
     */
    public int minMoves(int[] nums) {
        int num = Arrays.stream(nums).min().getAsInt();
        int res = 0;
        for(int i : nums){
            res += i - num;
        }
        return res;
    }

    /**
     * 假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。
     * 对每个孩子 i，都有一个胃口值g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j]。如果 s[j]>= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值
     * @param g
     * @param s
     * @return
     */
    public int findContentChildren(int[] g, int[] s) {
        int res = 0;
        Arrays.sort(g);
        Arrays.sort(s);
        int i = 0, j = 0;
        while(i < g.length && j < s.length){
            if(g[i] <= s[j]){
                res++;
                i++;
                j++;
            }else{
                j++;
            }
        }
        return res;
    }

    /**
     * 给定一个 row x col 的二维网格地图 grid ，其中：grid[i][j] = 1 表示陆地， grid[i][j] = 0 表示水域。
     * 网格中的格子 水平和垂直 方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，一个或多个表示陆地的格子相连组成的岛屿）。
     * 岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形，且宽度和高度均不超过 100 。计算这个岛屿的周长。
     * @param grid
     * @return
     */
    public int islandPerimeter(int[][] grid) {
        int res = 0;
        int row = grid.length;
        int col = grid[0].length;
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                if(grid[i][j] == 1){
                    res = res + 4;
                    if(0 <= i - 1 && grid[i - 1][j] == 1){
                        res--;
                    }
                    if(row > i + 1 && grid[i + 1][j] == 1){
                        res--;
                    }
                    if(0 <= j - 1 && grid[i][j - 1] == 1){
                        res--;
                    }
                    if(col > j + 1 && grid[i][j + 1] == 1){
                        res--;
                    }
                }
            }
        }
        return res;
    }

    /**
     * 给定一个二进制数组 nums ，计算其中最大连续 1 的个数。
     * @param nums
     */
    public int findMaxConsecutiveOnes(int[] nums) {
        int res = 0;
        int tmp = 0;
        for(int i : nums){
            if(i == 1){
                tmp++;
                res = Math.max(res, tmp);
            }else{
                tmp = 0;
            }
        }
        return res;
    }


    /**
     * nums1中数字x的 下一个更大元素 是指x在nums2 中对应位置 右侧 的 第一个 比x大的元素。
     * 给你两个 没有重复元素 的数组nums1 和nums2 ，下标从 0 开始计数，其中nums1是nums2的子集。
     * 对于每个 0 <= i < nums1.length ，找出满足 nums1[i] == nums2[j] 的下标 j ，并且在 nums2 确定 nums2[j] 的 下一个更大元素 。如果不存在下一个更大元素，那么本次查询的答案是 -1 。
     * 返回一个长度为nums1.length 的数组 ans 作为答案，满足 ans[i] 是如上所述的 下一个更大元素
     * @param nums1
     * @param nums2
     * @return
     */
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        int[] res = new int[nums1.length];
        Map<Integer, Integer> map = new HashMap<>();
        Stack<Integer> stack = new Stack<>();
        for(int i = nums2.length - 1; i >= 0; i--){
            int num = nums2[i];
            while(!stack.isEmpty() && num > stack.peek()){
                stack.pop();
            }
            map.put(num, stack.isEmpty() ? -1 : stack.peek());
            stack.push(num);
        }
        for(int i = 0; i < nums1.length; i++){
            res[i] = map.get(nums1[i]);
        }
        return res;
    }

    /**
     * 给你一个长度为 n 的整数数组 score ，其中 score[i] 是第 i 位运动员在比赛中的得分。所有得分都 互不相同 。
     * @param score
     * @return
     */
    public String[] findRelativeRanks(int[] score) {
        String[] res = new String[score.length];
        int[] sortedScore = score.clone();
        Arrays.sort(sortedScore);
        Map<Integer, String> map = new HashMap<>();
        for (int i = sortedScore.length - 1; i >= 0; i--) {
            if(i == sortedScore.length - 1){
                map.put(sortedScore[i], "Gold Medal");
            }else if(i == sortedScore.length - 2){
                map.put(sortedScore[i], "Silver Medal");
            }else if(i == sortedScore.length - 3){
                map.put(sortedScore[i], "Bronze Medal");
            }else{
                map.put(sortedScore[i], String.valueOf(sortedScore.length - i));
            }
        }
        for(int i = 0; i < score.length; i++){
            res[i] = map.get(score[i]);
        }
        return res;
    }

    public Integer findMinCoins(int sum, Map<Integer, Integer> map){
        if(map.containsKey(sum)){
            return map.get(sum);
        }
        int[] coins = new int[3];
        coins[0] = 2;
        coins[1] = 5;
        coins[2] = 6;
        int res = Integer.MAX_VALUE;
        if(sum == 0){
            return 0;
        }
        if(sum < 0){
            return -1;
        }
        for(int coin : coins){
            int sub = findMinCoins(sum - coin, map);
            if(sub == -1){
                continue;
            }
            res = Math.min(res, sub + 1);
            map.put(sum, res);
        }
        return res == Integer.MAX_VALUE ? -1 : res;
    }

    public Integer findMinCoins2(int sum){
        int[] dp = new int[sum + 1];
        for(int i = 0; i < dp.length; i++){
            dp[i] = sum + 1;
        }
        dp[0] = 0;
        int[] coins = new int[3];
        coins[0] = 2;
        coins[1] = 5;
        coins[2] = 6;
        for(int i = 0; i < dp.length; i++){
            for(int coin : coins){
                if(i - coin < 0){
                    continue;
                }
                dp[i] = Math.min(dp[i], 1 + dp[i - coin]);
            }
        }
        return dp[sum] == sum + 1 ? -1 : dp[sum];
    }

    private List<List<Integer>> res = Lists.newArrayList();

    public List<List<Integer>> permute(int[] nums){
        LinkedList<Integer> track = Lists.newLinkedList();
        backtrack(nums, track);
        return res;
    }

    private void backtrack(int[] nums, LinkedList<Integer> track){
        if(nums.length == track.size()){
            res.add(new LinkedList<>(track));
            return;
        }
        for(int i = 0; i < nums.length; i++){
            if(track.contains(nums[i])){
                continue;
            }
            track.add(nums[i]);
            backtrack(nums, track);
            //add是在后面add，所以remove也是在后面
            track.removeLast();
        }
    }

    private List<int[][]> nQueens = Lists.newArrayList();

    public int[][] solveNQueens(int n){
        int[][] board = new int[n][n];
        backtrack(board, 0);
        return null;
    }

    private void backtrack(int[][] nums, int row){
        if(nums.length == row){
            int[][] result = new int[row][row];
            for(int i = 0; i < row; i++){
                for(int j = 0; j < row; j++){
                    result[i][j] = nums[i][j];
                }
            }
            nQueens.add(result);
            return;
        }
        int n = nums[row].length;
        for(int col = 0; col < n; col++){
            if(!isValid(nums, row, col)){
                continue;
            }
            nums[row][col] = 1;
            backtrack(nums, row + 1);
            nums[row][col] = 0;
        }
    }

    private boolean isValid(int[][] nums, int row, int col){
        int n = nums.length;
        //检查同一列
        for(int i = 0; i < n; i++){
            if(nums[i][col] == 1){
                return false;
            }
        }
        //检查右上
        for(int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++){
            if(nums[i][j] == 1){
                return false;
            }
        }
        //检查左上
        for(int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--){
            if(nums[i][j] == 1){
                return false;
            }
        }
        return true;
    }

    public int minDepth(TreeNode root){
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.add(root);
        int depth = 1;
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i = 0; i < size; i++){
                TreeNode cur = queue.poll();
                if(cur.left == null && cur.right == null){
                    return depth;
                }
                if(cur.left != null){
                    queue.add(cur.left);
                }
                if(cur.right != null){
                    queue.add(cur.right);
                }
            }
            depth++;
        }
        return depth;
    }

    public int minLock(String target, String[] deadEnds){
        Queue<String> queue = new ArrayDeque<>();
        //记录死亡密码
        Set<String> deads = new HashSet<>();
        for(String dead : deadEnds){
            deads.add(dead);
        }
        //记录访问过的密码
        Set<String> visited = new HashSet<>();
        queue.add("0000");
        visited.add("0000");
        int res = 0;
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i = 0; i < size; i++){
                String cur = queue.poll();
                //跳过死亡密码
                if(deads.contains(cur)){
                    continue;
                }
                //终点条件
                if(cur.equals(target)){
                    return res;
                }
                for(int j = 0; j < 4; j++){
                    //向上转一位
                    String up = plusOne(cur, j);
                    if(!visited.contains(up)){
                        queue.add(up);
                        visited.add(up);
                    }
                    //向下转一位
                    String down = minusOne(cur, j);
                    if(!visited.contains(down)){
                        queue.add(down);
                        visited.add(down);
                    }
                }
            }
            //次数加一
            res++;
        }
        return -1;
    }

    private String minusOne(String s, int i){
        char[] c = s.toCharArray();
        if(c[i] == '0'){
            c[i] = '9';
        }else{
            c[i] -= 1;
        }
        return new String(c);
    }

    private String plusOne(String s, int i){
        char[] c = s.toCharArray();
        if(c[i] == '9'){
            c[i] = '0';
        }else{
            c[i] += 1;
        }
        return new String(c);
    }

    public int binarySearch(int[] nums, int target){
        int left = 0;
        int right = nums.length - 1;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if(nums[mid] == target){
                return mid;
            }else if (nums[mid] > target){
                right = mid - 1;
            }else if (nums[mid] < target){
                left = mid + 1;
            }
        }
        return -1;
    }

    public int left_bound(int[] nums, int target){
        int left = 0;
        int right = nums.length - 1;
        while (left <= right){
            int mid = left + (right - left) / 2;
            if(nums[mid] >= target){
                right = mid - 1;
            }else if (nums[mid] < target){
                left = mid + 1;
            }
        }
        if(left >= nums.length || nums[left] != target){
            return -1;
        }
        return left;
    }

    public int right_bound(int[] nums, int target){
        int left = 0;
        int right = nums.length - 1;
        while (left <= right){
            int mid = left + (right - left) / 2;
            if(nums[mid] > target){
                right = mid - 1;
            }else if (nums[mid] <= target){
                left = mid + 1;
            }
        }
        if(right < 0 || nums[right] != target){
            return -1;
        }
        return right;
    }

    /**
     * 用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，
     * 分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead操作返回 -1 )
     */
    Stack<Integer> inputStack = new Stack<>();
    Stack<Integer> outputStack = new Stack<>();
    public void cqueue(){

    }

    public void appendTail(Integer num){
        inputStack.push(num);
    }

    public Integer deleteHead(){
        if(outputStack.isEmpty()){
            if(inputStack.isEmpty()){
                return -1;
            }else{
                while(!inputStack.isEmpty()){
                    outputStack.push(inputStack.pop());
                }
            }
        }
        return outputStack.pop();
    }

    /**
     * 定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。
     */
    public void minStack(){
        Stack<Integer> stack = new Stack<>();
        Stack<Integer> minStack = new Stack<>();

        int num = 0;

        //push
        stack.push(num);
        if(minStack.isEmpty() || minStack.peek() >= num) {
            minStack.push(num);
        }

        //pop
        if(stack.isEmpty()){

        }else{

        }

        //min
        if(minStack.isEmpty()){

        }else {
            minStack.peek();
        }

        //top
        if(stack.isEmpty()){

        }else {
            stack.peek();
        }
    }

    /**
     * 输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
     * @param head
     * @return
     */
    public int[] reversePrint(ListNode head) {
        int sum = 0;
        List<Integer> tmp = new ArrayList<>();
        while (head != null){
            tmp.add(head.val);
            sum++;
            head = head.next;
        }
        int[] res = new int[sum];
        for(int i = tmp.size(); i > 0; i--){
            res[sum - i] = tmp.get(i - 1);
        }
        return res;
    }

    /**
     * 定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        ListNode next;
        ListNode pre = null;
        while (head != null) {
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }

    /**
     * 请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。
     * @param head
     * @return
     */
    public Node copyRandomList(Node head) {
        if(head == null) return null;
        Map<Node, Node> map = new HashMap<>();
        Node cur = head;
        while (cur != null){
            map.put(cur, new Node(cur.val));
            cur = cur.next;
        }
        cur = head;
        while (cur != null){
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
            cur = cur.next;
        }
        return map.get(head);
    }

    /**
     * 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
     * @param s
     * @return
     */
    public String replaceSpace(String s) {
        StringBuffer stringBuffer = new StringBuffer();
        for(char c : s.toCharArray()){
            if(c == ' '){
                stringBuffer.append("%20");
            }else {
                stringBuffer.append(c);
            }
        }
        return stringBuffer.toString();
    }

    /**
     * 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。
     * @param s
     * @param n
     * @return
     */
    public String reverseLeftWords(String s, int n) {
        int size = s.length();
        if(size < n || n < 0) return s;
        String res = s.substring(n) + s.substring(0, n);
        return res;
    }

    /**
     * 找出数组中重复的数字。
     * 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
     * @param nums
     * @return
     */
    public int findRepeatNumber(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for(int i : nums){
            if(set.contains(i)){
                return i;
            }
            set.add(i);
        }
        return -1;
    }

    /**
     * 统计一个数字在排序数组中出现的次数
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int i : nums){
            map.put(i, map.getOrDefault(i, 0) + 1);
        }
        return map.get(target) == null ? 0 : map.get(target);
    }

    /**
     * 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。
     * @param nums
     * @return
     */
    public int missingNumber_best(int[] nums) {
        int i = 0, j = nums.length - 1;
        while (i <= j){
            int mid = i + (j - i) / 2;
            if(nums[mid] == mid){
                i = mid + 1;
            }else{
                j = mid - 1;
            }
        }
        return i;
    }

    /**
     * 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
     * @param matrix
     * @param target
     * @return
     */
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if(matrix.length == 0 || matrix[0].length == 0) return false;
        int i = matrix.length - 1;
        int j = 0;
        while (i >= 0 && j <= matrix[0].length - 1){
            if(matrix[i][j] > target){
                i--;
            }else if(matrix[i][j] < target){
                j++;
            }else{
                return true;
            }
        }
        return false;
    }

    /**
     * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
     * 给你一个可能存在 重复 元素值的数组 numbers ，它原来是一个升序排列的数组，并按上述情形进行了一次旋转。请返回旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一次旋转，该数组的最小值为 1。
     * @param numbers
     * @return
     */
    public int minArray(int[] numbers) {
        int i = 0;
        int j = numbers.length - 1;
        while (i < j) {
            int m = (i + j) / 2;
            if (numbers[m] > numbers[j]) {
                i = m + 1;
            } else if (numbers[m] < numbers[j]) {
                j = m;
            } else {
                j--;
            }
        }
        return numbers[i];
    }

    /**
     * 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> queue = new ArrayDeque<>();
        List<List<Integer>> res = new ArrayList<>();
        if(root == null){
            return res;
        }
        queue.add(root);
        while (!queue.isEmpty()){
            List<Integer> tmp = new ArrayList<>();
            int sum = queue.size();
            int i = 0;
            while (!queue.isEmpty()){
                if(i == sum){
                    break;
                }
                TreeNode treeNode = queue.poll();
                tmp.add(treeNode.val);
                if(treeNode.left != null){
                    queue.add(treeNode.left);
                }
                if(treeNode.right != null){
                    queue.add(treeNode.right);
                }
                i++;
            }
            res.add(tmp);
        }
        return res;
    }

    public int[] levelOrder2(TreeNode root) {
        Queue<TreeNode> queue = new ArrayDeque<>();
        List<Integer> tmp = new ArrayList<>();
        if(root == null){
            return new int[0];
        }
        queue.add(root);
        while (!queue.isEmpty()){
            for(int i = queue.size(); i > 0; i--){
                TreeNode treeNode = queue.poll();
                tmp.add(treeNode.val);
                if(treeNode.left != null){
                    queue.add(treeNode.left);
                }
                if(treeNode.right != null){
                    queue.add(treeNode.right);
                }
            }
        }
        int[] res = new int[tmp.size()];
        for(int i = 0; i < tmp.size(); i++){
            res[i] = tmp.get(i);
        }
        return res;
    }

    public List<List<Integer>> levelOrder3(TreeNode root) {
        Queue<TreeNode> queue = new ArrayDeque<>();
        LinkedList<List<Integer>> res = new LinkedList<>();
        if(root == null){
            return res;
        }
        queue.add(root);
        int index = 0;
        while (!queue.isEmpty()){
            LinkedList<Integer> tmp = new LinkedList<>();
            for(int i = queue.size(); i > 0; i--){
                TreeNode treeNode = queue.poll();
                if(index % 2 == 0){
                    tmp.addLast(treeNode.val);
                }else {
                    tmp.addFirst(treeNode.val);
                }
                if(treeNode.left != null){
                    queue.add(treeNode.left);
                }
                if(treeNode.right != null){
                    queue.add(treeNode.right);
                }
            }
            res.add(tmp);
            index++;
        }
        return res;
    }

    /**
     * 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
     * B是A的子结构， 即 A中有出现和B相同的结构和节点值。
     * @param A
     * @param B
     * @return
     */
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if(A == null || B == null){
            return false;
        }
        return isSubStructureHelp(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B);
    }

    private boolean isSubStructureHelp(TreeNode A, TreeNode B){
        if(B == null){
            return true;
        }
        if(A == null || A.val != B.val){
            return false;
        }
        return (isSubStructureHelp(A.left, B.left) && isSubStructureHelp(A.right, B.right));
    }

    /**
     * 请完成一个函数，输入一个二叉树，该函数输出它的镜像。
     * @param root
     * @return
     */
    public TreeNode mirrorTree(TreeNode root) {
        mirrorTreeHelp(root);
        return root;
    }

    private void mirrorTreeHelp(TreeNode node){
        if((node != null) && (node.left != null || node.right != null)){
            TreeNode tmp = node.left;
            node.left = node.right;
            node.right = tmp;
        }else {
            return;
        }
        mirrorTreeHelp(node.left);
        mirrorTreeHelp(node.right);
    }

    /**
     * 请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。
     * @param root
     * @return
     */
    public boolean isSymmetric(TreeNode root) {
        if(root == null){
            return true;
        }
        return isSymmetricHelp(root.left, root.right);
    }

    private boolean isSymmetricHelp(TreeNode left, TreeNode right){
        if(left == null && right == null){
            return true;
        }
        if((left == null || right == null) || (left.val != right.val)){
            return false;
        }
        return isSymmetricHelp(left.left, right.right) && isSymmetricHelp(left.right, right.left);
    }

    /**
     * 写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下
     * F(0) = 0,   F(1) = 1
     * F(N) = F(N - 1) + F(N - 2), 其中 N > 1
     * @param n
     * @return
     */
    public int fib(int n) {
        if(n == 0){
            return 0;
        }
        if(n == 1){
            return 1;
        }
        int cur = 1;
        int pre = 0;
        while (n > 1){
            int tmp = pre;
            pre = cur % 1000000007;
            cur = (tmp + cur) % 1000000007;
            n--;
        }
        return cur % 1000000007;
    }

    /**
     * 假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？
     * @param prices
     * @return
     */
    public int maxProfit_best(int[] prices) {
        int cost = Integer.MAX_VALUE, profit = 0;
        for(int price : prices){
            cost = Math.min(cost, price);
            profit = Math.max(profit, price - cost);
        }
        return profit;
    }

    /**
     * 输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。
     * [-2,1,-3,4,-1,2,1,-5,4]
     * @param nums
     * @return
     */
    public int maxSubArray(int[] nums) {
        if(nums.length == 0) return 0;
        int res = nums[0];
        for(int i = 1; i < nums.length; i++){
            nums[i] += Math.max(nums[i-1], 0);
            res = Math.max(res, nums[i]);
        }
        return res;
    }

    /**
     * 在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？
     * [
     *   [1,3,1],
     *   [1,5,1],
     *   [4,2,1]
     * ]
     * @param grid
     * @return
     */
    public int maxValue(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(i == 0 && j == 0){
                    continue;
                }else if(i == 0){
                    grid[i][j] += grid[0][j - 1];
                }else if(j == 0){
                    grid[i][j] += grid[i - 1][j];
                }else {
                    grid[i][j] += Math.max(grid[i - 1][j], grid[i][j - 1]);
                }
            }
        }
        return grid[m - 1][n - 1];
    }

    /**
     * 请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度
     * abcabcbb
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        if (s == null) return 0;
        int i = 0, j = 0;
        int max = 0;
        Set<Character> set = new HashSet<>();
        char[] c = s.toCharArray();
        while(j < c.length){
            if(set.add(c[j])){
                j++;
                max = Math.max(max, j - i);
            }else{
                while(true){
                    set.remove(c[i]);
                    if(c[i] == c[j]){
                        i++;
                        break;
                    }
                    i++;
                }
            }
        }
        return max;

        //best
//        Map<Character,Integer> map = new HashMap<>();
//        int res = 0, tmp = 0;
//        for(int j = 0; j < s.length(); j++){
//            int i = map.getOrDefault(s.charAt(j), -1);
//            map.put(s.charAt(j), j);
//            tmp = tmp < j - i ? tmp + 1 : j - i;
//            res = Math.max(res, tmp);
//        }
//        return res;
    }

    /**
     * 输入两个链表，找出它们的第一个公共节点。
     * @param headA
     * @param headB
     * @return
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        Set<ListNode> listNodeSet = new HashSet<>();
        while (headA != null){
            listNodeSet.add(headA);
            headA = headA.next;
        }
        while (headB != null){
            if(listNodeSet.contains(headB)){
                return headB;
            }
            headB = headB.next;
        }
        return null;

        //best
//        ListNode A = headA, B = headB;
//        while(A != B){
//            A = A != null ? A.next : headB;
//            B = B != null ? B.next : headA;
//        }
//        return A;
    }

    /**
     * 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数在数组的前半部分，所有偶数在数组的后半部分。
     * 输入：nums = [1,2,3,4]
     * 输出：[1,3,2,4]
     * 注：[3,1,2,4] 也是正确的答案之一。
     * @param nums
     * @return
     */
    public int[] exchange(int[] nums) {
        int i = 0, j = 0;
        while (j < nums.length){
            if(nums[j] % 2 == 1){
                int tmp = nums[i];
                nums[i] = nums[j];
                nums[j] = tmp;
                i++;
            }
            j++;
        }
        return nums;
    }

    /**
     * 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
     * 输入：nums = [2,7,11,15], target = 9
     * 输出：[2,7] 或者 [7,2]
     * @param nums
     * @param target
     * @return
     */
    public int[] twoSum_best(int[] nums, int target) {
        int i = 0, j = nums.length - 1;
        while(i < j){
            int sum = nums[i] + nums[j];
            if (sum < target) {
                i++;
            } else if (sum > target) {
                j--;
            } else {
                return new int[]{nums[i], nums[j]};
            }
        }
        return new int[0];
    }

    /**
     * 输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。
     * 输入: "the sky is blue"
     * 输出: "blue is sky the"
     * "  hello world!  "
     * @param s
     * @return
     */
    public String reverseWords(String s) {
        s = s.trim();
        String[] strings = s.split(" ");
        String res = "";
        for(int i = strings.length - 1; i >= 0 ; i--){
            if(strings[i] == ""){
                continue;
            }
            res += strings[i];
            if(i != 0){
                res += " ";
            }
        }
        return res;
    }

    /**
     * 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
     * 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        Set<String> visited = new HashSet<>();
        for (int i = 0; i < board.length; i++){
            for (int j = 0; j < board[0].length; j++){
                if(word.charAt(0) != board[i][j]){
                    continue;
                }
                visited.add((i + "") + (j + ""));
                if(exist_help(board, word, visited, i, j, 1)){
                    return true;
                }
                visited.clear();
            }
        }
        return false;
    }

    private boolean exist_help(char[][] board, String word, Set<String> visited, int i, int j, int k){
        if(word.length() == k){
            return true;
        }
        boolean up = false;
        boolean down = false;
        boolean left = false;
        boolean right = false;
        if(i - 1 >= 0 && !visited.contains((i - 1 + "") + (j + "")) && board[i - 1][j] == word.charAt(k)){
            visited.add((i - 1 + "") + (j + ""));
            up = exist_help(board, word, visited, i - 1, j, ++k);
            visited.remove((i - 1 + "") + (j + ""));
            k--;
        }
        if(up){
            return true;
        }
        if(i + 1 < board.length && !visited.contains((i + 1 + "") + (j + "")) && board[i + 1][j] == word.charAt(k)){
            visited.add((i + 1 + "") + (j + ""));
            down = exist_help(board, word, visited, i + 1, j, ++k);
            visited.remove((i + 1 + "") + (j + ""));
            k--;
        }
        if(down){
            return true;
        }
        if(j - 1 >= 0 && !visited.contains((i + "") + (j - 1 + "")) && board[i][j - 1] == word.charAt(k)){
            visited.add((i + "") + (j - 1 + ""));
            left = exist_help(board, word, visited, i, j - 1, ++k);
            visited.remove((i + "") + (j - 1 + ""));
            k--;
        }
        if(left){
            return true;
        }
        if(j + 1 < board[0].length && !visited.contains((i + "") + (j + 1 + "")) && board[i][j + 1] == word.charAt(k)){
            visited.add((i + "") + (j + 1 + ""));
            right = exist_help(board, word, visited, i, j + 1, ++k);
            visited.remove((i + "") + (j + 1 + ""));
            k--;
        }
        if(right){
            return true;
        }
        return false;

        //best
//        if(i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != word.charAt(k)){
//            return false;
//        }
//        if(k == word.length()-1){
//            return true;
//        }
//        char temp = board[i][j];
//        board[i][j] = '/';
//        boolean res = (dfs(board,word,i-1,j,k+1) || dfs(board,word,i+1,j,k+1) || dfs(board,word,i,j-1,k+1) || dfs(board,word,i,j+1,k+1));
//        board[i][j] = temp;
//        return res;
    }

    /**
     * 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），
     * 也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。
     * 但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？
     * @param m
     * @param n
     * @param k
     * @return
     */
    public int movingCount(int m, int n, int k) {
        int res = 0;
        int[][] board = new int[m][n];
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(movingCount_help(board, i, j ,k)){
                    res++;
                }
            }
        }
        return res;
    }

    private boolean movingCount_help(int[][] board, int i, int j, int k){
        int sum = 0;
        int m = i;
        int n = j;
        while (m > 0){
            sum += m % 10;
            m = m / 10;
        }
        while (n > 0){
            sum += n % 10;
            n = n / 10;
        }
        if(sum > k){
            return false;
        }
        if(i - 1 < 0){
            if(j - 1 < 0){
                board[i][j] = 1;
                return true;
            }else if(board[i][j - 1] == 1){
                board[i][j] = 1;
                return true;
            }else{
                return false;
            }
        }else if(j - 1 < 0){
            if(i - 1 < 0){
                board[i][j] = 1;
                return true;
            }else if(board[i - 1][j] == 1){
                board[i][j] = 1;
                return true;
            }else {
                return false;
            }
        }else if(board[i - 1][j] == 1 || board[i][j - 1] == 1){
            board[i][j] = 1;
            return true;
        }else {
            return false;
        }
    }

    /**
     * 给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
     *
     * 叶子节点 是指没有子节点的节点。
     * @param root
     * @param target
     * @return
     */
    LinkedList<Integer> path = new LinkedList<>();
    public List<List<Integer>> pathSum(TreeNode root, int target) {
        pathSum_help(root, target);
        return res;
    }

    private void pathSum_help(TreeNode root, int target){
        if(root == null){
            return;
        }
        path.add(root.val);
        target -= root.val;
        if(root.left == null && root.right == null && target == 0){
            res.add(new LinkedList<>(path));
            path.removeLast();
            return;
        }
        pathSum_help(root.left, target);
        pathSum_help(root.right, target);
        path.removeLast();
    }

    /**
     * 输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
     * @param nums
     * @return
     */
    public String minNumber(int[] nums) {
        String[] strings = new String[nums.length];
        for(int i = 0; i < nums.length; i++){
            strings[i] = String.valueOf(nums[i]);
        }
        minNumber_help(strings, 0, nums.length - 1);
        StringBuffer stringBuffer = new StringBuffer();
        for(String s : strings){
            stringBuffer.append(s);
        }
        return stringBuffer.toString();
    }

    //快速排序
    private void minNumber_help(String[] strings, int l, int r){
        if(r <= l){
            return;
        }
        int i = l;
        int j = r;
        String tmp = strings[i];
        while (i < j){
            //从后往前找比tmp小的数
            while (i < j && (strings[j] + strings[l]).compareTo(strings[l] + strings[j]) >= 0){
                j--;
            }
            //从前往后找比tmp大的数
            while (i < j && (strings[i] + strings[l]).compareTo(strings[l] + strings[i]) <= 0){
                i++;
            }
            //交换i、j指针
            tmp = strings[i];
            strings[i] = strings[j];
            strings[j] = tmp;
        }
        //交换第一个数与i指针的数，第i位为tmp
        strings[i] = strings[l];
        strings[l] = tmp;
        //i + 1 后面的比tmp大
        minNumber_help(strings, i + 1, r);
        //i - 1 后面的比tmp小
        minNumber_help(strings, l, i - 1);
    }

    /**
     * 从若干副扑克牌中随机抽 5 张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，
     * A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。
     * @param nums
     * @return
     */
    public boolean isStraight(int[] nums) {
        Set<Integer> set = new HashSet<>();
        int max = 0;
        int min = 14;
        for(int num : nums){
            if(num == 0){
                continue;
            }
            if(set.contains(num)){
                return false;
            }
            set.add(num);
            max = Math.max(max, num);
            min = Math.min(min, num);
        }
        return max - min < 5;
    }

    /**
     * 输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
     * @param arr
     * @param k
     * @return
     */
    public int[] getLeastNumbers(int[] arr, int k) {
        Arrays.sort(arr);
        int[] res = new int[k];
        for(int i = 0; i < k ;i++){
            res[i] = arr[i];
        }
        return res;
    }

    //best
    private int[] getLeastNumbers_help(int[] arr, int l, int r, int k){
        int random = (int) Math.random() * (r - l + 1) + l;
        swap(arr, random, l);
        int i = l, j = r;
        while (i < j) {
            while (i < j && arr[j] >= arr[l]) {
                j--;
            }
            while (i < j && arr[i] <= arr[l]) {
                i++;
            }
            swap(arr, i, j);
        }
        swap(arr, l, i);
        if (k < i) {
            return getLeastNumbers_help(arr, k, l, i - 1);
        }
        if (k > i) {
            return getLeastNumbers_help(arr, k, i + 1, r);
        }
        return Arrays.copyOf(arr, k);
    }

    /**
     * 输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if(root == null){
            return 0;
        }
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    /**
     * 输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。
     * @param root
     * @return
     */
    public boolean isBalanced(TreeNode root) {
        int res = isBalanced_help(root);
        return res != -1;
    }

    private int isBalanced_help(TreeNode root){
        if(root == null){
            return 0;
        }
        int left = isBalanced_help(root.left);
        if(left == -1){
            return -1;
        }
        int right = isBalanced_help(root.right);
        if(right == -1){
            return -1;
        }
        return Math.abs(left - right) < 2 ? Math.max(left, right) + 1 : -1;
    }

    /**
     * 求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
     * @param n
     * @return
     */
    public int sumNums(int n) {
        boolean x = n > 0 && (n += sumNums(n - 1)) > 0;
        return n;
    }

    /**
     * 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
     * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，
     * 最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）”
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(p.val > root.val && q.val > root.val){
            return lowestCommonAncestor(root.right, p, q);
        }else if(p.val < root.val && q.val < root.val){
            return lowestCommonAncestor(root.left, p, q);
        }else {
            return root;
        }
    }

    /**
     * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
     * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，
     * 最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）”
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || root == p || root == q){
            return root;
        }
        TreeNode left = lowestCommonAncestor2(root.left, p, q);
        TreeNode right = lowestCommonAncestor2(root.right, p, q);
        if(left == null){
            return right;
        }
        if(right == null){
            return left;
        }
        return root;
    }

    /**
     * 输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。
     * 假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
     * preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        Map<Integer, Integer> inorderMap = new HashMap<>();
        for(int i = 0; i < inorder.length; i++){
            inorderMap.put(inorder[i], i);
        }
        TreeNode res = buildTree_help(preorder, inorderMap, 0, 0, inorder.length - 1);
        return res;
    }

    private TreeNode buildTree_help(int[] preorder, Map<Integer, Integer> inorderMap, int root, int left, int right){
        if(left > right) {
            return null;
        }
        TreeNode node = new TreeNode(preorder[root]);
        int i = inorderMap.get(preorder[root]);
        node.left = buildTree_help(preorder, inorderMap, root + 1, left, i - 1);
        //i - left为左子树长度，root为起始位置，下一个即是右子树的根节点再加1
        node.right = buildTree_help(preorder, inorderMap, root + i - left + 1, i + 1, right);
        return node;
    }

    /**
     * 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。
     * [1,3,2,6,5]
     * @param postorder
     * @return
     */
    public boolean verifyPostorder(int[] postorder) {
        return verifyPostorder_help(postorder, 0, postorder.length - 1);
    }

    private boolean verifyPostorder_help(int[] postorder, int left, int right){
        if(left >= right){
            return true;
        }
        int p = left;
        while (postorder[p] < postorder[right]){
            p++;
        }
        int m = p;
        while (postorder[p] > postorder[right]){
            p++;
        }
        return p == right && verifyPostorder_help(postorder, left, m - 1) && verifyPostorder_help(postorder, m, right - 1);
    }

    /**
     * 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
     * 请你将两个数相加，并以相同形式返回一个表示和的链表。
     * 你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
     * 输入：l1 = [2,4,3], l2 = [5,6,4]
     * 输出：[7,0,8]
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode res = new ListNode(0);
        ListNode head = new ListNode(0);
        res.next = head;
        boolean flag = false;
        while (l1 != null && l2 != null){
            int val = l1.val + l2.val;
            if(flag){
                val++;
                flag = false;
            }
            if(val >= 10){
                val = val % 10;
                flag = true;
            }
            head.next = new ListNode(val);
            head = head.next;
            l1 = l1.next;
            l2 = l2.next;
        }
        while (l1 != null){
            int val = l1.val;
            if(flag){
                val++;
                flag = false;
            }
            if(val >= 10){
                val = val % 10;
                flag = true;
            }
            head.next = new ListNode(val);
            head = head.next;
            l1 = l1.next;
        }
        while (l2 != null){
            int val = l2.val;
            if(flag){
                val++;
                flag = false;
            }
            if(val >= 10){
                val = val % 10;
                flag = true;
            }
            head.next = new ListNode(val);
            head = head.next;
            l2 = l2.next;
        }
        if(flag){
            head.next = new ListNode(1);
        }
        return res.next.next;
    }

    /**
     * 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
     * abcaecbb
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring_best(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int tmp = 0;
        int res = 0;
        for(int j = 0; j < s.length(); j++){
            int i = map.getOrDefault(s.charAt(j), -1);
            map.put(s.charAt(j), j);
            // s.charAt[j]和s.charAt[i]，如果i在上一个char的最大字串长度（tmp）内，则s.charAt[j]的最大字串长度是j - i，
            // 如果如果i在上一个char的最大字串长度（tmp）外，则则s.charAt[j]的最大字串长度是tmp + 1
            tmp = j - i > tmp ? tmp + 1 : j - i;
            res = Math.max(res, tmp);
        }
        return res;
    }

    /**
     * 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
     * 算法的时间复杂度应该为 O(log (m+n)) 。
     * @param nums1
     * @param nums2
     * @return
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        double res = 0;
        int i = 0;
        int j = 0;
        while (i < m || j < n){
            double tmp;
            if(i >= m){
                tmp = nums2[j];
                j++;
            }else if(j >= n){
                tmp = nums1[i];
                i++;
            }else if(nums1[i] > nums2[j]){
                tmp = nums2[j];
                j++;
            }else {
                tmp = nums1[i];
                i++;
            }
            if((m + n) % 2 == 1){
                if((m + n) / 2 + 1 == i + j){
                    return tmp;
                }
            }else {
                if((m + n) / 2 == i + j){
                    res += tmp;
                }
                if((m + n) / 2 + 1 == i + j){
                    res += tmp;
                    return (double)res / 2;
                }
            }
        }
        return res;
    }

    /**
     * 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，
     * 其中 B[i] 的值是数组 A 中除了下标 i 以外的元素的积, 即 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。
     * @param a
     * @return
     */
    public int[] constructArr(int[] a) {
        if(a.length == 0) {
            return new int[0];
        }
        int[] b = new int[a.length];
        b[0] = 1;
        for(int i = 1; i < a.length; i++){
            b[i] = a[i - 1] * b[i - 1];
        }
        int tmp = 1;
        for(int i = a.length - 2; i >= 0; i--){
            tmp *= a[i + 1];
            b[i] *= tmp;
        }
        return b;
    }

    /**
     * 给你一个字符串 s，找到 s 中最长的回文子串。
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        char[] chars = s.toCharArray();
        int length = chars.length;
        boolean[][] dp = new boolean[length][length];
        //单个字符串为回文子串
        for(int i = 0; i < length; i++){
            dp[i][i] = true;
        }
        int begin = 0;
        int maxLength = 1;
        //确定步长，2步长的只要确保两个字符相等即可，l可以是字符串最大长度
        for(int l = 2; l <= length; l++){
            for(int i = 0; i < length - 1; i++){
                //i为起始下标，j通过i + l - 1计算，即起始下标加步长减1
                int j = l + i - 1;
                if(j >= length){
                    break;
                }
                if(chars[i] != chars[j]){
                    dp[i][j] = false;
                }else {
                    //j - i只有两种情况，等于1时就是只有两个字符，相等即是回文；等于2时就是三个字符，只要判断i、j两个字符是否相等
                    if(j - i < 3){
                        dp[i][j] = true;
                    }else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                if(dp[i][j] && j - i + 1 > maxLength){
                    maxLength = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxLength);
    }

    /**
     * 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。
     * 请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
     * @param n
     * @return
     */
    public int cuttingRope(int n) {
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 1;
        for(int i = 3; i <= n; i++){
            for(int j = 1; j < i; j++){
                //第一种情况是剪两段，j为一段，i - j为一段；第二种情况是剪三段，j为一段，dp[i - j]是前面的最大值，可能是一段也可能是两段
                int tmp = Math.max(j * (i - j), j * dp[i - j]);
                dp[i] = Math.max(dp[i], tmp);
            }
        }
        return dp[n];
    }

    /**
     * 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
     * 序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
     * @param target
     * @return
     */
    public int[][] findContinuousSequence(int target) {
        int i = 1;
        int j = 2;
        int s = 3;
        List<int[]> list = new LinkedList<>();
        while (i < j){
            if(s == target){
                int[] tmp = new int[j - i + 1];
                for(int k = 0; k <= j - i; k++){
                    tmp[k] = i + k;
                }
                list.add(tmp);
            }
            if(s >= target){
                s -= i;
                i++;
            }else {
                j++;
                s += j;
            }
        }
        return list.toArray(new int[0][]);
    }

    /**
     * 0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。
     * 例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。
     * @param n
     * @param m
     * @return
     */
    public int lastRemaining(int n, int m) {
        int x = 0;
        for(int i = 2; i <= n; i++){
            x = (x + m) % i;
        }
        return x;
    }

    /**
     * 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。
     * 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
     * 输出：[1,2,3,6,9,8,7,4,5]
     * @param matrix
     * @return
     */
    public int[] spiralOrder(int[][] matrix) {
        if(matrix.length == 0){
            return new int[0];
        }
        int[] res = new int[matrix.length * matrix[0].length];
        int sum = 0;
        int left = 0, right = matrix[0].length - 1, up = 0, down = matrix.length - 1;
        while (true){
            for(int i = left; i <= right; i++){
                res[sum++] = matrix[up][i];
            }
            if(++up > down){
                break;
            }

            for(int i = up; i <= down; i++){
                res[sum++] = matrix[i][right];
            }
            if(--right < left){
                break;
            }

            for(int i = right; i >= left; i--){
                res[sum++] = matrix[down][i];
            }
            if(--down < up){
                break;
            }

            for(int i = down; i >= up; i--){
                res[sum++] = matrix[i][left];
            }
            if(++left > right){
                break;
            }
        }
        return res;
    }

    /**
     * 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。
     * 例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。
     * @param pushed
     * @param popped
     * @return
     */
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        if(pushed.length == 0){
            return true;
        }
        int index = 0;
        Stack<Integer> stack = new Stack<>();
        for(int i = 0; i < pushed.length; i++){
            stack.push(pushed[i]);
            while (!stack.isEmpty() && index < popped.length && stack.peek() == popped[index]){
                stack.pop();
                index++;
            }
        }
        return stack.isEmpty();
    }

    /**
     * 给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
     * 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
     * 返回容器可以储存的最大水量。
     * @param height
     * @return
     */
    public int maxArea2(int[] height) {
        int res = 0;
        int l = 0, r = height.length - 1;
        while (l < r){
            res = Math.max(res, (r - l) * Math.min(height[l], height[r]));
            if(height[l] >= height[r]){
                r--;
            }else {
                l++;
            }
        }
        return res;
    }

    /**
     * 给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。
     * 输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
     * 输出: [3,3,5,5,6,7]
     * @param nums
     * @param k
     * @return
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums == null || k == 0){
            return new int[0];
        }
        int[] res = new int[nums.length - k + 1];
        Deque<Integer> deque = new ArrayDeque<>();
        for(int i = 1 - k, j = 0; j < nums.length; i++, j++){
            if (i > 0 && !deque.isEmpty() && deque.peekFirst() == nums[i - 1]){
                deque.pollFirst();
            }
            while(!deque.isEmpty() && deque.peekLast() < nums[j]){
                deque.pollLast();
            }
            deque.addLast(nums[j]);
            if(i >= 0){
                res[i] = deque.peekFirst();
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
     * @param nums
     * @return
     */
    public List<List<Integer>> threeSum2(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if(nums == null || nums.length < 3){
            return res;
        }
        Arrays.sort(nums);
        for(int i = 0; i < nums.length - 2; i++){
            if(nums[i] > 0){
                break;
            }
            if(i > 0 && nums[i - 1] == nums[i]){
                continue;
            }
            int l = i + 1;
            int r = nums.length - 1;
            int target = -nums[i];
            while (l < r){
                if(target - nums[r] == nums[l]){
                    res.add(new ArrayList<>(Arrays.asList(nums[i], nums[l], nums[r])));
                    l++;
                    r--;
                    while (l < r && nums[l - 1] == nums[l]){
                        l++;
                    }
                    while (l < r && nums[r + 1] == nums[r]){
                        r--;
                    }
                }else if(target - nums[r] > nums[l]){
                    l++;
                }else {
                    r--;
                }
            }
        }
        return res;
    }

    /**
     *输入一个字符串，打印出该字符串中字符的所有排列。
     * 你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。
     * 输入：s = "abc"
     * 输出：["abc","acb","bac","bca","cab","cba"]
     * @param s
     * @return
     */
    Set<String> set = new HashSet<>();
    public String[] permutation(String s) {
        permutation_help(s.toCharArray(), new StringBuffer(), new boolean[s.length()]);
        return set.toArray(new String[0]);
    }

    private void permutation_help(char[] chars, StringBuffer stringBuffer, boolean[] flag){
        if(chars.length == stringBuffer.length()){
            set.add(stringBuffer.toString());
            return;
        }
        for(int i = 0; i < chars.length; i++){
            if(flag[i]){
                continue;
            }
            stringBuffer.append(chars[i]);
            flag[i] = true;
            permutation_help(chars, stringBuffer, flag);
            stringBuffer.deleteCharAt(stringBuffer.length() - 1);
            flag[i] = false;
        }
    }

    /**
     * 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
     * @param digits
     * @return
     */
    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if(digits == null || digits.length() == 0){
            return res;
        }
        Map<Character, String> map = new HashMap<>();
        map.put('2', "abc");
        map.put('3', "def");
        map.put('4', "ghi");
        map.put('5', "jkl");
        map.put('6', "mno");
        map.put('7', "pqrs");
        map.put('8', "tuv");
        map.put('9', "wxyz");
        letterCombinations_help(map, digits, new StringBuffer(), 0);
        return res;
    }

    private void letterCombinations_help(Map<Character, String> map, String digits, StringBuffer stringBuffer, int index){
        if(index == stringBuffer.length()){
//            res.add(stringBuffer.toString());
            return;
        }
        String str = map.get(digits.charAt(index));
        for(int i = 0; i < str.length(); i++){
            stringBuffer.append(str.charAt(i));
            letterCombinations_help(map, digits, stringBuffer, index + 1);
            stringBuffer.deleteCharAt(stringBuffer.length() - 1);
        }
    }

    /**
     * 我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。
     * @param n
     * @return
     */
    public int nthUglyNumber(int n) {
        int a = 0, b = 0, c = 0;
        int[] db = new int[n];
        db[0] = 1;
        for(int i = 1; i < n; i++){
            int n2 = db[a] * 2;
            int n3 = db[b] * 3;
            int n5 = db[c] * 5;
            db[i] = Math.min(Math.min(n2, n3), n5);
            if(db[i] == n2){
                a++;
            }
            if(db[i] == n3){
                b++;
            }
            if(db[i] == n5){
                c++;
            }
        }
        return db[n - 1];
    }

    /**
     * 把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
     * 你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。
     * @param n
     * @return
     */
    public double[] dicesProbability(int n) {
        double[] res = new double[6];
        Arrays.fill(res, 1 / 6.0);
        for(int i = 2; i <= n; i++){
            double[] tmp = new double[5 * i + 1];
            for(int j = 0; j < res.length; j++){
                for(int k = 0; k < 6; k++){
                    tmp[j + k] += res[j] / 6.0;
                }
            }
            res = tmp;
        }
        return res;
    }

    /**
     * 输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。
     * @param n
     * @return
     */
    public int[] printNumbers(int n) {
        int[] res = new int[(int)Math.pow(10, n) - 1];
        for(int i = 0; i < res.length; i++){
            res[i] = i + 1;
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
        if(head == null){
            return null;
        }
        int size = 0;
        ListNode node = head;
        while(node != null){
            size++;
            node = node.next;
        }
        if(n == size){
            return head.next;
        }
        node = head;
        for(int i = 0; i < size - n - 1; i++){
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
     * 每个右括号都有一个对应的相同类型的左括号
     * @param s
     * @return
     */
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for(char c : s.toCharArray()){
            if(c == '(' || c == '{' || c == '['){
                stack.push(c);
            }else{
                if(stack.isEmpty()){
                    return false;
                }else if((c == ')' && stack.peek() == '(') ||
                        (c == '}' && stack.peek() == '{') ||
                        (c == ']' && stack.peek() == '[')){
                    stack.pop();
                }
                else{
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }

    /**
     * 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
     * @param list1
     * @param list2
     * @return
     */
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode node = new ListNode();
        ListNode head = node;
        while (list1 != null && list2 != null){
            if(list1.val <= list2.val){
                node.next = list1;
                list1 = list1.next;
            }else {
                node.next = list2;
                list2 = list2.next;
            }
            node = node.next;
        }
        if(list1 != null){
            node.next = list1;
        }
        if(list2 != null){
            node.next = list2;
        }
        return head.next;
    }

    /**
     * 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
     * @param n
     * @return
     */
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        generateParenthesis_help(res, new StringBuffer(), n, 0, 0);
        return res;
    }

    private void generateParenthesis_help(List<String> res, StringBuffer stringBuffer, int n, int left, int right){
        if(stringBuffer.length() == n * 2){
            res.add(stringBuffer.toString());
            return;
        }
        if(left < n){
            stringBuffer.append("(");
            generateParenthesis_help(res, stringBuffer, n, left + 1, right);
            stringBuffer.deleteCharAt(stringBuffer.length() - 1);
        }
        if(right < left){
            stringBuffer.append(")");
            generateParenthesis_help(res, stringBuffer, n, left, right + 1);
            stringBuffer.deleteCharAt(stringBuffer.length() - 1);
        }
    }

    /**
     * 给你一个链表数组，每个链表都已经按升序排列。
     * 请你将所有链表合并到一个升序链表中，返回合并后的链表。
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists == null || lists.length == 0){
            return null;
        }
        ListNode node = new ListNode();
        ListNode head = node;
        node.next = lists[0];
        for(int i = 1; i < lists.length; i++){
            ListNode list1 = head.next;
            ListNode list2 = lists[i];
            while (list1 != null && list2 != null){
                if(list1.val <= list2.val){
                    node.next = list1;
                    list1 = list1.next;
                }else {
                    node.next = list2;
                    list2 = list2.next;
                }
                node = node.next;
            }
            if(list1 != null){
                node.next = list1;
            }
            if(list2 != null){
                node.next = list2;
            }
            node = head;
        }
        return head.next;
    }

    /**
     * 整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，
     * 如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。
     * 如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。
     * 1 2 4 3
     * @param nums
     */
    public void nextPermutation(int[] nums) {
        int i = 0, j = nums.length - 1;
        while (j > 0){
            if(nums[j] <= nums[j - 1]){
                j--;
            }else {
                i = j - 1;
                break;
            }
        }
        if(j == 0){
            nextPermutation_help(nums, 0, nums.length - 1);
        }else {
            for(int k = nums.length - 1; k > j; k--){
                if(nums[k] > nums[i]){
                    j = k;
                }
            }
            swap(nums, i, j);
            nextPermutation_help(nums, i + 1, nums.length - 1);
        }
    }

    private void nextPermutation_help(int[] nums, int i, int j){
        while (i < j){
            int tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
            i++;
            j--;
        }
    }

    /**
     * 给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
     * 输入：s = ")()())"
     * 输出：4
     * ()(()
     * (()
     * (()(((()
     * @param s
     * @return
     */
    public int longestValidParentheses(String s) {
        int max = 0;
        int left = 0;
        int right = 0;
        for(int i = 0; i < s.length(); i++){
            if(s.charAt(i) == '('){
                left++;
            }else {
                right++;
            }
            if(left == right){
                max = Math.max(max, (right * 2));
            }else if(left < right){
                left = right = 0;
            }
        }
        left = right = 0;
        for(int i = s.length() - 1; i >= 0; i--){
            if(s.charAt(i) == ')'){
                right++;
            }else {
                left++;
            }
            if(left == right){
                max = Math.max(max, (left * 2));
            }else if(left > right){
                left = right = 0;
            }
        }
        return max;
    }

    /**
     * 整数数组 nums 按升序排列，数组中的值 互不相同 。
     * 在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，
     * 使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。
     * 例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
     * 给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
     * 你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
     * [7,0,1,2,4,5,6]
     * @param nums
     * @param target
     * @return
     */
    public int search2(int[] nums, int target) {
        int i = 0, j = nums.length - 1;
        while (i <= j){
            int mid = i + (j - i) / 2;
            if(nums[mid] == target){
                return mid;
            }
            //这样要用等号保证左边是包括自己在内是升序
            if(nums[i] <= nums[mid]){
                if(nums[mid] > target && nums[i] <= target){
                    j = mid - 1;
                }else {
                    i = mid + 1;
                }
            }else {
                if(nums[mid] < target && nums[j] >= target){
                    i = mid + 1;
                }else {
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
     * @param nums
     * @param target
     * @return
     */
    public int[] searchRange(int[] nums, int target) {
        int i = 0, j = nums.length - 1;
        while (i <= j){
            int mid = i + (j - i) / 2;
            if(nums[mid] >= target){
                j = mid - 1;
            }else {
                i = mid + 1;
            }
        }
        if(i >= nums.length || nums[i] != target){
            return new int[]{-1, -1};
        }else {
            int[] res = new int[2];
            res[0] = i;
            while (i + 1 < nums.length && nums[i + 1] == target){
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
     * 对于给定的输入，保证和为 target 的不同组合数少于 150 个。
     * @param candidates
     * @param target
     * @return
     */
    List<List<Integer>> combinationSum_Res = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        combinationSum_help(new LinkedList<>(), candidates, target, 0, 0);
        return combinationSum_Res;
    }

    private void combinationSum_help(LinkedList<Integer> tmp, int[] candidates, int target, int sum, int start){
        if(sum > target){
            return;
        }
        if(sum == target){
            combinationSum_Res.add(new ArrayList<>(tmp));
            return;
        }
        for(int i = start; i < candidates.length; i++){
            tmp.addLast(candidates[i]);
            combinationSum_help(tmp, candidates, target, sum + candidates[i], i);
            tmp.removeLast();
        }
    }

    /**
     * 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
     * [0,1,0,2,1,0,1,3,2,1,2,1]
     * @param height
     * @return
     */
    public int trap(int[] height) {
        int res = 0;
        int[] leftMax = new int[height.length];
        int[] rightMax = new int[height.length];
        leftMax[0] = height[0];
        for(int i = 1; i < height.length; i++){
            leftMax[i] = Math.max(leftMax[i - 1], height[i]);
        }
        rightMax[height.length - 1] = height[height.length - 1];
        for(int i = height.length - 2; i >= 0; i--){
            rightMax[i] = Math.max(rightMax[i + 1], height[i]);
        }
        for(int i = 0; i < height.length; i++){
            res += Math.min(leftMax[i], rightMax[i]) - height[i];
        }
        return res;
    }

    /**
     * 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
     * @param nums
     * @return
     */
    List<List<Integer>> permute_res = new ArrayList<>();
    public List<List<Integer>> permute1(int[] nums) {
        permute_help(new LinkedList<>(), nums);
        return permute_res;
    }

    private void permute_help(LinkedList<Integer> tmp, int[] nums){
        if(tmp.size() == nums.length){
            permute_res.add(new ArrayList<>(tmp));
        }
        for(int num : nums){
            if(tmp.contains(num)){
                continue;
            }
            tmp.add(num);
            permute_help(tmp, nums);
            tmp.removeLast();
        }
    }

    /**
     * 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
     *
     * 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        int[][] tmp = new int[matrix.length][matrix[0].length];
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[0].length; j++){
                //旋转后的规律是原来的第i行第j列的值放到新的第j行倒数第i列
                tmp[j][matrix.length - i - 1] = matrix[i][j];
            }
        }
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[0].length; j++){
                matrix[i][j] = tmp[i][j];
            }
        }
    }

    public void rotate_best(int[][] matrix) {
        int size = matrix.length;
        //先上下翻转
        for(int i = 0; i < size / 2; i++){
            for(int j = 0; j < size; j++){
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[size - i - 1][j];
                matrix[size - i - 1][j] = tmp;
            }
        }
        //再对角线翻转
        for(int i = 0; i < size; i++){
            for(int j = i + 1; j < size; j++){
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
        for(String str : strs){
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            if(map.containsKey(new String(chars))){
                map.get(new String(chars)).add(str);
            }else {
                List<String> tmp = new ArrayList<>();
                tmp.add(str);
                map.put(new String(chars), tmp);
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
     * 解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
     * @param nums
     * @return
     */
    public int maxSubArray2(int[] nums) {
        int max = nums[0];
        for(int i = 1; i < nums.length; i++){
            nums[i] += Math.max(nums[i - 1], 0);
            max = Math.max(max, nums[i]);
        }
        return max;
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
        int size = nums.length;
        //获取当前下标能达到的最大位置
        int rightMax = 0;
        for(int i = 0; i < size; i++){
            //判断是否能到达下一个位置，如果i > rightMax即是前面的最大位置达不到当前i的位置
            if(i <= rightMax){
                rightMax = Math.max(rightMax, i + nums[i]);
                //如果能直接跳跃到最大长度则直接返回true
                if(rightMax >= size - 1){
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
     */
    public int[][] merge(int[][] intervals) {
        if(intervals == null || intervals.length == 0){
            return new int[0][0];
        }
        //根据左区间排序
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        List<int[]> list = new ArrayList<>();
        for(int i = 0; i < intervals.length; i++){
            int left = intervals[i][0], right = intervals[i][1];
            if(list.isEmpty() || list.get(list.size() - 1)[1] < left){
                //list为空或者下一个的左区间大于上一个的右区间，直接插入
                list.add(new int[]{left, right});
            }else {
                //对比右区间取最大值
                list.get(list.size() - 1)[1] = Math.max(list.get(list.size() - 1)[1], right);
            }
        }
        return list.toArray(new int[list.size()][]);
    }

    /**
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
     * 问总共有多少条不同的路径？
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        int[][] tmp = new int[m][n];
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(i == 0 || j == 0){
                    tmp[i][j] = 1;
                    continue;
                }
                if(i - 1 >= 0){
                    tmp[i][j] += tmp[i - 1][j];
                }
                if(j - 1 >= 0){
                    tmp[i][j] += tmp[i][j - 1];
                }
            }
        }
        return tmp[m - 1][n - 1];
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
        for(int i = 1; i < m; i++){
            grid[i][0] += grid[i - 1][0];
        }
        for(int i = 1; i < n; i++){
            grid[0][i] += grid[0][i - 1];
        }
        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        return grid[m - 1][n - 1];
    }

    /**
     * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
     * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        if(n == 1){
            return 1;
        }
        if(n == 2){
            return 2;
        }
        int cur = 2;
        int pre = 1;
        while (n > 2){
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
     * 必须在不使用库的sort函数的情况下解决这个问题。
     * 输入：nums = [2,0,2,1,1,0]
     * 输出：[0,0,1,1,2,2]
     * @param nums
     */
    public void sortColors(int[] nums) {
        sortColors_help(nums, 0, nums.length - 1);
    }

    private void sortColors_help(int[] nums, int l, int r){
        if(l >= r){
            return;
        }
        int index = sortColors_help_quick(nums, l, r);
        sortColors_help(nums, l, index - 1);
        sortColors_help(nums, index + 1, r);
    }

    private int sortColors_help_quick(int[] nums, int l, int r){
        int k = nums[l];
        int i = l;
        int j = r;
        while (i < j){
            while (i < j && k < nums[j]){
                j--;
            }
            while (i < j && k >= nums[i]){
                i++;
            }
            if(i < j){
                swap(nums, i, j);
            }
        }
        nums[l] = nums[j];
        nums[j] = k;
        return j;
    }

    /**
     * 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
     * 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
     * 输入：nums = [1,2,3]
     * 输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
     * @param nums
     * @return
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> tmp = new ArrayList<>();
        subsets_help(nums, res, tmp, 0);
        return res;
    }

    private void subsets_help(int[] nums, List<List<Integer>> res, List<Integer> tmp, int start){
        res.add(new ArrayList<>(tmp));
        if(start == nums.length){
            return;
        }
        for(int i = start; i < nums.length; i++){
            tmp.add(nums[i]);
            subsets_help(nums, res, tmp, i + 1);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
     * 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
     * @param board
     * @param word
     * @return
     */
    public boolean exist1(char[][] board, String word) {
        int m = board.length;
        int n = board[0].length;
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(exist_help_best(board, word, i, j, 0)){
                    return true;
                }
            }
        }
        return false;
    }

    private boolean exist_help_best(char[][] board, String word, int i, int j, int k){
        if(i < 0 || i > board.length - 1 || j < 0 || j > board[0].length - 1 || board[i][j] != word.charAt(k)){
            return false;
        }
        if(k == word.length() - 1){
            return true;
        }
        char tmp = board[i][j];
        board[i][j] = '/';
        boolean res = exist_help_best(board, word, i + 1, j, k + 1)
                || exist_help_best(board, word, i - 1, j, k + 1)
                || exist_help_best(board, word, i, j + 1, k + 1)
                || exist_help_best(board, word, i, j - 1, k + 1);
        board[i][j] = tmp;
        return res;
    }

    /**
     * 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
     * 求在该柱状图中，能够勾勒出来的矩形的最大面积。
     * [2,1,5,6,2,3] [-1,-1,1,2,1,4] [1,6,4,4,6,6]
     * 10
     * @param heights
     * @return
     */
    public int largestRectangleArea(int[] heights) {
        int res = 0;
        //单调栈
        Stack<Integer> stack = new Stack<>();
        int n = heights.length;
        //找出左边比当前下标小的最近的下标
        int[] left = new int[n];
        //找出右边比当前下标小的最近下标
        int[] right = new int[n];
        for(int i = 0; i < n; i++){
            //等号也弹出
            while (!stack.isEmpty() && heights[i] <= heights[stack.peek()]){
                stack.pop();
            }
            //如果单调栈空了则存边界外的值
            left[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }
        stack.clear();
        for(int i = n - 1; i >= 0; i--){
            while (!stack.isEmpty() && heights[i] <= heights[stack.peek()]){
                stack.pop();
            }
            right[i] = stack.isEmpty() ? n : stack.peek();
            stack.push(i);
        }
        for(int i = 0; i < n; i++){
            //记得减一
            res = Math.max(res, (right[i] - left[i] - 1) * heights[i]);
        }
        return res;
    }

    /**
     * 给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。
     * @param matrix
     * @return
     */
    public int maximalRectangle(char[][] matrix) {
        int res = 0;
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] left = new int[m][n];
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                if (matrix[i][j] == '1') {
                    left[i][j] = (j == 0 ? 0 : left[i][j - 1]) + 1;
                }
            }
        }
        //基于每一列进行柱状图求最大面积
        for (int j = 0; j < n; j++){
            int[] up = new int[m];
            int[] down = new int[m];
            Stack<Integer> stack = new Stack<>();
            for (int i = 0; i < m; i++){
                while (!stack.isEmpty() && left[stack.peek()][j] >= left[i][j]){
                    stack.pop();
                }
                up[i] = stack.isEmpty() ? -1 : stack.peek();
                stack.push(i);
            }
            stack.clear();
            for (int i = m - 1; i >= 0; i--){
                while (!stack.isEmpty() && left[stack.peek()][j] >= left[i][j]){
                    stack.pop();
                }
                down[i] = stack.isEmpty() ? m : stack.peek();
                stack.push(i);
            }
            for (int i = 0; i < m; i++){
                res = Math.max(res, (down[i] - up[i] - 1) * left[i][j]);
            }
        }
        return res;
    }

    /**
     * 给定一个二叉树的根节点 root ，返回 它的 中序 遍历
     * @param root
     * @return
     */
    List<Integer> inorderTraversal_res = new ArrayList<>();
    public List<Integer> inorderTraversal(TreeNode root) {
        if(root == null){
            return new ArrayList<>();
        }
        inorderTraversal(root.left);
        inorderTraversal_res.add(root.val);
        inorderTraversal(root.right);
        return inorderTraversal_res;
    }

    /**
     * 给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？
     * 返回满足题意的二叉搜索树的种数。
     * @param n
     * @return
     */
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++){
            for (int j = 1; j <= i; j++){
                //假设除了根节点左边的，就有i-j个节点
                int left = dp[i - j];
                //假设除了根节点右边的，就有j-1个节点
                int right = dp[j - 1];
                //两边相乘再相加就是当前节点数的个数
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
        isValidBST_help(root);
        for(int i = 1; i < isValidBST_res.size(); i++){
            if(isValidBST_res.get(i) <= isValidBST_res.get(i - 1)){
                return false;
            }
        }
        return true;
    }

    private List<Integer> isValidBST_help(TreeNode root){
        if(root == null){
            return new ArrayList<>();
        }
        isValidBST_help(root.left);
        isValidBST_res.add(root.val);
        isValidBST_help(root.right);
        return isValidBST_res;
    }

    /**
     * 给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder1(TreeNode root) {
        if (root == null){
            return new ArrayList<>();
        }
        Queue<TreeNode> queue = new ArrayDeque<>();
        List<List<Integer>> res = new ArrayList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            int sum = queue.size();
            List<Integer> tmp = new ArrayList<>();
            for(int i = 0; i < sum; i++){
                TreeNode node = queue.poll();
                tmp.add(node.val);
                if(node.left != null){
                    queue.add(node.left);
                }
                if(node.right != null){
                    queue.add(node.right);
                }
            }
            res.add(tmp);
        }
        return res;
    }

    public static void main(String[] args) {
        Algorithm algorithm = new Algorithm();
        int[] pushed = new int[]{1,2,3,3};
        System.out.println(algorithm.largestRectangleArea(new int[]{2,1,5,6,2,3}));
    }
}
