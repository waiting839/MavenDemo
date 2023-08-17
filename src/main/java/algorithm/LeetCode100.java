package algorithm;

import java.util.*;

/**
 * @author 吴嘉烺
 * @description
 * @date 2023/8/17
 */
public class LeetCode100 {

    /**
     * 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
     * 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
     * 你可以按任意顺序返回答案。
     * 输入：nums = [2,7,11,15], target = 9
     * 输出：[0,1]
     * 解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
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
        return new int[0];
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
        List<List<String>> res = new ArrayList<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            List<String> list = map.getOrDefault(new String(chars), new ArrayList<>());
            list.add(str);
            map.put(new String(chars), list);
        }
        map.forEach((k, v) -> res.add(v));
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
            if (!set.contains(num + 1)) {
                int cur = num;
                int curLen = 1;
                while (set.contains(cur - 1)) {
                    cur--;
                    curLen++;
                }
                res = Math.max(res, curLen);
            }
        }
        return res;
    }

    public static void main(String[] args) {
        LeetCode100 leetCode100 = new LeetCode100();
        leetCode100.groupAnagrams(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"});
    }
}
