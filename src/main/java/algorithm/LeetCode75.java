package algorithm;

import com.google.common.primitives.Chars;

import java.util.*;

/**
 * @author 吴嘉烺
 * @description
 * @date 2023/7/14
 */
public class LeetCode75 {

    /**
     * 给你两个字符串 word1 和 word2 。请你从 word1 开始，通过交替添加字母来合并字符串。
     * 如果一个字符串比另一个字符串长，就将多出来的字母追加到合并后字符串的末尾。
     * 返回 合并后的字符串 。
     * 输入：word1 = "abc", word2 = "pqr"
     * 输出："apbqcr"
     * 解释：字符串合并情况如下所示：
     * word1：  a   b   c
     * word2：    p   q   r
     * 合并后：  a p b q c r
     * @param word1
     * @param word2
     * @return
     */
    public String mergeAlternately(String word1, String word2) {
        StringBuilder stringBuilder = new StringBuilder();
        int i = 0;
        int j = 0;
        while (i < word1.length() && j < word2.length()) {
            stringBuilder.append(word1.charAt(i++));
            stringBuilder.append(word2.charAt(j++));
        }
        if (i < word1.length()) {
            stringBuilder.append(word1.substring(i));
        }
        if (j < word2.length()) {
            stringBuilder.append(word2.substring(j));
        }
        return stringBuilder.toString();
    }

    /**
     * 对于字符串 s 和 t，只有在 s = t + ... + t（t 自身连接 1 次或多次）时，我们才认定 “t 能除尽 s”。
     * 给定两个字符串 str1 和 str2 。返回 最长字符串 x，要求满足 x 能除尽 str1 且 x 能除尽 str2 。
     * 输入：str1 = "ABCABC", str2 = "ABC"
     * 输出："ABC"
     * @param str1
     * @param str2
     * @return
     */
    public String gcdOfStrings(String str1, String str2) {
        //如果都能除尽，str1 + str2 会等于 str2 + str1（字符串拼接）
        if (!str1.concat(str2).equals(str2.concat(str1))) {
            return "";
        }
        return str1.substring(0, gcdOfStrings_help(str1.length(), str2.length()));
    }

    /**
     * 找出a,b之间的最大公约数
     * @param a
     * @param b
     * @return
     */
    private int gcdOfStrings_help(int a, int b) {
        int tmp = a % b;
        while (tmp != 0) {
            a = b;
            b = tmp;
            tmp = a % b;
        }
        return b;
    }

    /**
     * 给你一个数组 candies 和一个整数 extraCandies ，其中 candies[i] 代表第 i 个孩子拥有的糖果数目。
     * 对每一个孩子，检查是否存在一种方案，将额外的 extraCandies 个糖果分配给孩子们之后，此孩子有 最多 的糖果。
     * 注意，允许有多个孩子同时拥有 最多 的糖果数目。
     * 输入：candies = [2,3,5,1,3], extraCandies = 3
     * 输出：[true,true,true,false,true]
     * 解释：
     * 孩子 1 有 2 个糖果，如果他得到所有额外的糖果（3个），那么他总共有 5 个糖果，他将成为拥有最多糖果的孩子。
     * 孩子 2 有 3 个糖果，如果他得到至少 2 个额外糖果，那么他将成为拥有最多糖果的孩子。
     * 孩子 3 有 5 个糖果，他已经是拥有最多糖果的孩子。
     * 孩子 4 有 1 个糖果，即使他得到所有额外的糖果，他也只有 4 个糖果，无法成为拥有糖果最多的孩子。
     * 孩子 5 有 3 个糖果，如果他得到至少 2 个额外糖果，那么他将成为拥有最多糖果的孩子。
     * @param candies
     * @param extraCandies
     * @return
     */
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        List<Boolean> res = new ArrayList<>();
        int max = 0;
        for (int num : candies) {
            max = Math.max(max, num);
        }
        for (int num : candies) {
            res.add(num + extraCandies >= max);
        }
        return res;
    }

    /**
     * 假设有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花不能种植在相邻的地块上，它们会争夺水源，两者都会死去。
     * 给你一个整数数组 flowerbed 表示花坛，由若干 0 和 1 组成，其中 0 表示没种植花，1 表示种植了花。另有一个数 n ，
     * 能否在不打破种植规则的情况下种入 n 朵花？能则返回 true ，不能则返回 false 。
     * 输入：flowerbed = [1,0,0,0,1], n = 1
     * 输出：true
     * @param flowerbed
     * @param n
     * @return
     */
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int max = 0;
        int countZero = 0;
        for (int i = 0; i < flowerbed.length; i++) {
            if (i == 0 && flowerbed[i] == 0) {
                countZero++;
            }
            if (i == flowerbed.length - 1 && flowerbed[i] == 0) {
                countZero++;
            }
            if (flowerbed[i] == 0) {
                countZero++;
            } else {
                if (countZero >= 3) {
                    max += 1 + (countZero - 3) / 2;
                    if (max >= n) {
                        return true;
                    }
                }
                countZero = 0;
            }
        }
        if (countZero >= 3) {
            max += 1 + (countZero - 3) / 2;
        }
        return max >= n;
    }

    /**
     * 给你一个字符串 s ，仅反转字符串中的所有元音字母，并返回结果字符串。
     * 元音字母包括 'a'、'e'、'i'、'o'、'u'，且可能以大小写两种形式出现不止一次。
     * 输入：s = "hello"
     * 输出："holle"
     * @param s
     * @return
     */
    public String reverseVowels(String s) {
        char[] chars = s.toCharArray();
        int i = 0;
        int j = s.length() - 1;
        while (i < j) {
            while (i < s.length() - 1 &&!isVowels(chars[i])) {
                i++;
            }
            while (j > 0 && !isVowels(chars[j])) {
                j--;
            }
            if (i < j) {
                char tmp = chars[i];
                chars[i] = chars[j];
                chars[j] = tmp;
                i++;
                j--;
            }
        }
        return new String(chars);
    }

    private boolean isVowels(char c) {
        return "aeiouAEIOU".indexOf(c) >= 0;
    }

    /**
     * 给你一个字符串 s ，请你反转字符串中 单词 的顺序。
     * 单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。
     * 返回 单词 顺序颠倒且 单词 之间用单个空格连接的结果字符串。
     * 注意：输入字符串 s中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。
     * 输入：s = "the sky is blue"
     * 输出："blue is sky the"
     * @param s
     * @return
     */
    public String reverseWords(String s) {
        s = s.trim();
        String[] strings = s.split(" ");
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = strings.length - 1; i >= 0; i--) {
            if (strings[i].equals("")) {
                continue;
            }
            stringBuilder.append(strings[i]).append(" ");
        }
        stringBuilder.deleteCharAt(stringBuilder.length() - 1);
        return stringBuilder.toString();
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
        for (int i = 0; i < nums.length; i++) {
            res[i] = p;
            p *= nums[i];
        }
        for (int i = nums.length - 1; i >= 0; i--) {
            res[i] *= q;
            q *= nums[i];
        }
        return res;
    }

    /**
     * 给你一个整数数组 nums ，判断这个数组中是否存在长度为 3 的递增子序列。
     * 如果存在这样的三元组下标 (i, j, k) 且满足 i < j < k ，使得 nums[i] < nums[j] < nums[k] ，返回 true ；否则，返回 false 。
     * 输入：nums = [1,2,3,4,5]
     * 输出：true
     * 解释：任何 i < j < k 的三元组都满足题意
     * @param nums
     * @return
     */
    public boolean increasingTriplet(int[] nums) {
        int n = nums.length;
        if (n < 3) {
            return false;
        }
        int first = nums[0];
        int second = Integer.MAX_VALUE;
        for (int i = 1; i < n; i++) {
            int num = nums[i];
            if (num > second) {
                return true;
            } else if (num > first) {
                second = num;
            } else {
                first = num;
            }
        }
        return false;
    }

    /**
     * 给你一个字符数组 chars ，请使用下述算法压缩：
     * 从一个空字符串 s 开始。对于 chars 中的每组 连续重复字符 ：
     * 如果这一组长度为 1 ，则将字符追加到 s 中。
     * 否则，需要向 s 追加字符，后跟这一组的长度。
     * 压缩后得到的字符串 s 不应该直接返回 ，需要转储到字符数组 chars 中。需要注意的是，如果组长度为 10 或 10 以上，则在 chars 数组中会被拆分为多个字符。
     * 请在 修改完输入数组后 ，返回该数组的新长度。
     * 你必须设计并实现一个只使用常量额外空间的算法来解决此问题。
     * 输入：chars = ["a","b","b","b","b","b","b","b","b","b","b","b","b"]
     * 输出：返回 4 ，输入数组的前 4 个字符应该是：["a","b","1","2"]。
     * 解释：由于字符 "a" 不重复，所以不会被压缩。"bbbbbbbbbbbb" 被 “b12” 替代。
     * @param chars
     * @return
     */
    public int compress(char[] chars) {
        //记录修改的下标
        int write = 0;
        //记录不同字符的下标的起点
        int left = 0;
        for (int i = 0; i < chars.length; i++) {
            if (i == chars.length - 1 || chars[i] != chars[i + 1]) {
                chars[write] = chars[i];
                write++;
                //计算相同字符的长度
                int num = i - left + 1;
                if (num > 1) {
                    int start = write;
                    while (num > 0) {
                        chars[write] = (char) (num % 10 + '0');
                        write++;
                        num /= 10;
                    }
                    reverse(chars, start, write - 1);
                }
                left = i + 1;
            }
        }
        return write;
    }

    public void reverse(char[] chars, int left, int right) {
        while (left < right) {
            char temp = chars[left];
            chars[left] = chars[right];
            chars[right] = temp;
            left++;
            right--;
        }
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
                int tmp = nums[i];
                nums[i] = nums[j];
                nums[j] = tmp;
                j++;
            }
        }
    }

    /**
     * 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
     * 字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（
     * 例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
     * 进阶：
     * 如果有大量输入的 S，称作 S1, S2, ... , Sk 其中 k >= 10亿，你需要依次检查它们是否为 T 的子序列。在这种情况下，你会怎样改变代码？
     * 输入：s = "abc", t = "ahbgdc"
     * 输出：true
     * @param s
     * @param t
     * @return
     */
    public boolean isSubsequence(String s, String t) {
        if (s.length() == 0) {
            return true;
        }
        int i = 0;
        int j = 0;
        while (j < t.length()) {
            if (s.charAt(i) == t.charAt(j)) {
                i++;
                if (i == s.length()) {
                    return true;
                }
            }
            j++;
        }
        return false;
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
        int l = 0;
        int r = height.length - 1;
        while (l < r) {
            res = Math.max(res, Math.min(height[l], height[r]) * (r - l));
            if (height[l] > height[r]) {
                r--;
            } else {
                l++;
            }
        }
        return res;
    }

    /**
     * 给你一个整数数组 nums 和一个整数 k 。
     * 每一步操作中，你需要从数组中选出和为 k 的两个整数，并将它们移出数组。
     * 返回你可以对数组执行的最大操作数。
     * 输入：nums = [1,2,3,4], k = 5
     * 输出：2
     * 解释：开始时 nums = [1,2,3,4]：
     * - 移出 1 和 4 ，之后 nums = [2,3]
     * - 移出 2 和 3 ，之后 nums = []
     * 不再有和为 5 的数对，因此最多执行 2 次操作。
     * @param nums
     * @param k
     * @return
     */
    public int maxOperations(int[] nums, int k) {
        Arrays.sort(nums);
        int i = 0;
        int j = nums.length - 1;
        int res = 0;
        while (i < j) {
            if (k - nums[j] == nums[i]) {
                res++;
                i++;
                j--;
            } else if (k - nums[j] > nums[i]) {
                i++;
            } else {
                j--;
            }
        }
        return res;
    }

    /**
     * 给你一个由 n 个元素组成的整数数组 nums 和一个整数 k 。
     * 请你找出平均数最大且 长度为 k 的连续子数组，并输出该最大平均数。
     * 任何误差小于 10^-5 的答案都将被视为正确答案。
     * 输入：nums = [1,12,-5,-6,50,3], k = 4
     * 输出：12.75
     * 解释：最大平均数 (12-5-6+50)/4 = 51/4 = 12.75
     * @param nums
     * @param k
     * @return
     */
    public double findMaxAverage(int[] nums, int k) {
        double sum = 0;
        for (int i = 0; i < k; i++) {
            sum += nums[i];
        }
        double maxSum = sum;
        for (int i = k; i < nums.length; i++) {
            sum -= nums[i - k];
            sum += nums[i];
            maxSum = Math.max(maxSum, sum);
        }
        return maxSum / k;
    }

    /**
     * 给你字符串 s 和整数 k 。
     * 请返回字符串 s 中长度为 k 的单个子字符串中可能包含的最大元音字母数。
     * 英文中的 元音字母 为（a, e, i, o, u）。
     * 输入：s = "abciiidef", k = 3
     * 输出：3
     * 解释：子字符串 "iii" 包含 3 个元音字母。
     * @param s
     * @param k
     * @return
     */
    public int maxVowels(String s, int k) {
        char[] chars = s.toCharArray();
        int sum = 0;
        for (int i = 0; i < k; i++) {
            if (isVowels(chars[i])) {
                sum++;
            }
        }
        if (sum == k) {
            return sum;
        }
        int res = sum;
        for (int i = k; i < s.length(); i++) {
            if (isVowels(chars[i - k])) {
                sum--;
            }
            if (isVowels(chars[i])) {
                sum++;
            }
            if (sum == k) {
                return sum;
            }
            res = Math.max(res, sum);
        }
        return res;
    }

    /**
     * 给定一个二进制数组 nums 和一个整数 k，如果可以翻转最多 k 个 0 ，则返回 数组中连续 1 的最大个数 。
     * 输入：nums = [1,1,1,0,0,0,1,1,1,1,0], K = 2
     * 输出：6
     * 解释：[1,1,1,0,0,1,1,1,1,1,1]
     * 粗体数字从 0 翻转到 1，最长的子数组长度为 6。
     * @param nums
     * @param k
     * @return
     */
    public int longestOnes(int[] nums, int k) {
        int left = 0;
        //左下标之前0的个数
        int lSum = 0;
        //右下标之前0的个数
        int rSum = 0;
        int res = 0;
        for (int right = 0; right < nums.length; right++) {
            rSum += 1 - nums[right];
            //保证区间[left, right]之间的0个数小于等于k即可
            while (lSum < rSum - k) {
                lSum += 1 - nums[left];
                left++;
            }
            res = Math.max(res, right - left + 1);
        }
        return res;
    }

    /**
     * 给你一个二进制数组 nums ，你需要从中删掉一个元素。
     * 请你在删掉元素的结果数组中，返回最长的且只包含 1 的非空子数组的长度。
     * 如果不存在这样的子数组，请返回 0
     * 输入：nums = [1,1,0,1]
     * 输出：3
     * 解释：删掉位置 2 的数后，[1,1,1] 包含 3 个 1 。
     * @param nums
     * @return
     */
    public int longestSubarray(int[] nums) {
        int left = 0;
        //左下标之前0的个数
        int lSum = 0;
        //右下标之前0的个数
        int rSum = 0;
        int res = 0;
        for (int right = 0; right < nums.length; right++) {
            rSum += 1 - nums[right];
            while (lSum < rSum - 1) {
                lSum += 1 - nums[left];
                left++;
            }
            res = Math.max(res, right - left);
        }
        return res;
    }

    /**
     * 有一个自行车手打算进行一场公路骑行，这条路线总共由 n + 1 个不同海拔的点组成。自行车手从海拔为 0 的点 0 开始骑行。
     * 给你一个长度为 n 的整数数组 gain ，其中 gain[i] 是点 i 和点 i + 1 的 净海拔高度差（0 <= i < n）。请你返回 最高点的海拔 。
     * 输入：gain = [-5,1,5,0,-7]
     * 输出：1
     * 解释：海拔高度依次为 [0,-5,-4,1,1,-6] 。最高海拔为 1 。
     * @param gain
     * @return
     */
    public int largestAltitude(int[] gain) {
        int res = 0;
        int cur = 0;
        for (int num : gain) {
            cur += num;
            res = Math.max(res, cur);
        }
        return res;
    }

    /**
     * 给你一个整数数组 nums ，请计算数组的 中心下标 。
     * 数组 中心下标 是数组的一个下标，其左侧所有元素相加的和等于右侧所有元素相加的和。
     * 如果中心下标位于数组最左端，那么左侧数之和视为 0 ，因为在下标的左侧不存在元素。这一点对于中心下标位于数组最右端同样适用。
     * 如果数组有多个中心下标，应该返回 最靠近左边 的那一个。如果数组不存在中心下标，返回 -1 。
     * 输入：nums = [1, 7, 3, 6, 5, 6]
     * 输出：3
     * 解释：
     * 中心下标是 3 。
     * 左侧数之和 sum = nums[0] + nums[1] + nums[2] = 1 + 7 + 3 = 11 ，
     * 右侧数之和 sum = nums[4] + nums[5] = 5 + 6 = 11 ，二者相等。
     * @param nums
     * @return
     */
    public int pivotIndex(int[] nums) {
        int lSum = 0;
        int rSum = 0;
        for (int i = 1; i < nums.length; i++) {
            rSum += nums[i];
        }
        if (lSum == rSum) {
            return 0;
        }
        for (int i = 1; i < nums.length; i++) {
            lSum += nums[i - 1];
            rSum -= nums[i];
            if (lSum == rSum) {
                return i;
            }
        }

        return -1;
    }

    /**
     * 给你两个下标从 0 开始的整数数组 nums1 和 nums2 ，请你返回一个长度为 2 的列表 answer ，其中：
     * answer[0] 是 nums1 中所有 不 存在于 nums2 中的 不同 整数组成的列表。
     * answer[1] 是 nums2 中所有 不 存在于 nums1 中的 不同 整数组成的列表。
     * 注意：列表中的整数可以按 任意 顺序返回。
     * 输入：nums1 = [1,2,3], nums2 = [2,4,6]
     * 输出：[[1,3],[4,6]]
     * 解释：
     * 对于 nums1 ，nums1[1] = 2 出现在 nums2 中下标 0 处，然而 nums1[0] = 1 和 nums1[2] = 3 没有出现在 nums2 中。因此，answer[0] = [1,3]。
     * 对于 nums2 ，nums2[0] = 2 出现在 nums1 中下标 1 处，然而 nums2[1] = 4 和 nums2[2] = 6 没有出现在 nums2 中。因此，answer[1] = [4,6]。
     * @param nums1
     * @param nums2
     * @return
     */
    public List<List<Integer>> findDifference(int[] nums1, int[] nums2) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path1 = new ArrayList<>();
        List<Integer> path2 = new ArrayList<>();
        Set<Integer> set1 = new HashSet<>();
        Set<Integer> set2 = new HashSet<>();
        for (int num : nums1) {
            set1.add(num);
        }
        for (int num : nums2) {
            set2.add(num);
        }
        for (int num : set1) {
            if (!set2.contains(num)) {
                path1.add(num);
            }
        }
        for (int num : set2) {
            if (!set1.contains(num)) {
                path2.add(num);
            }
        }
        res.add(path1);
        res.add(path2);
        return res;
    }

    /**
     * 给你一个整数数组 arr，请你帮忙统计数组中每个数的出现次数。
     * 如果每个数的出现次数都是独一无二的，就返回 true；否则返回 false。
     * 输入：arr = [1,2,2,1,1,3]
     * 输出：true
     * 解释：在该数组中，1 出现了 3 次，2 出现了 2 次，3 只出现了 1 次。没有两个数的出现次数相同。
     * @param arr
     * @return
     */
    public boolean uniqueOccurrences(int[] arr) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : arr) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        Set<Integer> set = new HashSet<>();
        map.forEach((k, v) -> set.add(v));
        return set.size() == map.size();
    }

    /**
     * 如果可以使用以下操作从一个字符串得到另一个字符串，则认为两个字符串 接近 ：
     * 操作 1：交换任意两个 现有 字符。
     * 例如，abcde -> aecdb
     * 操作 2：将一个 现有 字符的每次出现转换为另一个 现有 字符，并对另一个字符执行相同的操作。
     * 例如，aacabb -> bbcbaa（所有 a 转化为 b ，而所有的 b 转换为 a ）
     * 你可以根据需要对任意一个字符串多次使用这两种操作。
     * 给你两个字符串，word1 和 word2 。如果 word1 和 word2 接近 ，就返回 true ；否则，返回 false 。
     * 输入：word1 = "abc", word2 = "bca"
     * 输出：true
     * 解释：2 次操作从 word1 获得 word2 。
     * 执行操作 1："abc" -> "acb"
     * 执行操作 1："acb" -> "bca"
     * @param word1
     * @param word2
     * @return
     */
    public boolean closeStrings(String word1, String word2) {
        if (word1.length() != word2.length()) {
            return false;
        }
        int n = word1.length();
        int[] int1 = new int[26];
        int[] int2 = new int[26];
        for (int i = 0; i < n; i++) {
            int1[word1.charAt(i) - 'a']++;
            int2[word2.charAt(i) - 'a']++;
        }
        for (int i = 0; i < int1.length; i++) {
            if (int1[i] == 0 && int2[i] > 0) {
                return false;
            }
            if (int2[i] == 0 && int1[i] > 0) {
                return false;
            }
        }
        Arrays.sort(int1);
        Arrays.sort(int2);
        for (int i = 0; i < int1.length; i++) {
            if (int1[i] != int2[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * 给你一个下标从 0 开始、大小为 n x n 的整数矩阵 grid ，返回满足 Ri 行和 Cj 列相等的行列对 (Ri, Cj) 的数目。
     * 如果行和列以相同的顺序包含相同的元素（即相等的数组），则认为二者是相等的。
     * 输入：grid = [[3,2,1],[1,7,6],[2,7,7]]
     * 输出：1
     * 解释：存在一对相等行列对：
     * - (第 2 行，第 1 列)：[2,7,7]
     * @param grid
     * @return
     */
    public int equalPairs(int[][] grid) {
        int res = 0;
        Map<String, Integer> row = new HashMap<>();
        Map<String, Integer> col = new HashMap<>();
        int n = grid.length;
        for (int i = 0; i < n; i++) {
            StringBuilder rowStr = new StringBuilder();
            StringBuilder colStr = new StringBuilder();
            for (int j = 0; j < n; j++) {
                rowStr.append(grid[i][j]).append(",");
                colStr.append(grid[j][i]).append(",");
            }
            row.put(rowStr.toString(), row.getOrDefault(rowStr.toString(), 0) + 1);
            col.put(colStr.toString(), col.getOrDefault(colStr.toString(), 0) + 1);
        }
        for (Map.Entry e : row.entrySet()) {
            res += (int) e.getValue() * col.getOrDefault(e.getKey(), 0);
        }
        return res;
    }

    /**
     * 给你一个包含若干星号 * 的字符串 s 。
     * 在一步操作中，你可以：
     * 选中 s 中的一个星号。
     * 移除星号 左侧 最近的那个 非星号 字符，并移除该星号自身。
     * 返回移除 所有 星号之后的字符串。
     * 注意：
     * 生成的输入保证总是可以执行题面中描述的操作。
     * 可以证明结果字符串是唯一的。
     * 输入：s = "leet**cod*e"
     * 输出："lecoe"
     * 解释：从左到右执行移除操作：
     * - 距离第 1 个星号最近的字符是 "leet**cod*e" 中的 't' ，s 变为 "lee*cod*e" 。
     * - 距离第 2 个星号最近的字符是 "lee*cod*e" 中的 'e' ，s 变为 "lecod*e" 。
     * - 距离第 3 个星号最近的字符是 "lecod*e" 中的 'd' ，s 变为 "lecoe" 。
     * 不存在其他星号，返回 "lecoe" 。
     * @param s
     * @return
     */
    public String removeStars(String s) {
        StringBuilder stringBuilder = new StringBuilder();
        Deque<Character> deque = new ArrayDeque<>();
        for (char c : s.toCharArray()) {
            if (c == '*') {
                deque.pop();
            } else {
                deque.push(c);
            }
        }
        while (!deque.isEmpty()) {
            stringBuilder.append(deque.pollLast());
        }
        return stringBuilder.toString();
    }

    /**
     * 给定一个整数数组 asteroids，表示在同一行的行星。
     * 对于数组中的每一个元素，其绝对值表示行星的大小，正负表示行星的移动方向（正表示向右移动，负表示向左移动）。每一颗行星以相同的速度移动。
     * 找出碰撞后剩下的所有行星。碰撞规则：两个行星相互碰撞，较小的行星会爆炸。如果两颗行星大小相同，则两颗行星都会爆炸。
     * 两颗移动方向相同的行星，永远不会发生碰撞。
     * 输入：asteroids = [5,10,-5]
     * 输出：[5,10]
     * 解释：10 和 -5 碰撞后只剩下 10 。 5 和 10 永远不会发生碰撞。
     * @param asteroids
     * @return
     */
    public int[] asteroidCollision(int[] asteroids) {
        Deque<Integer> deque = new ArrayDeque<>();
        for (int num : asteroids) {
            boolean boom = false;
            if (!deque.isEmpty() && (deque.peekFirst() > 0 && num < 0)) {
                while (!deque.isEmpty() && (deque.peekFirst() > 0 && num < 0)) {
                    if (deque.peekFirst() == -num) {
                        deque.pollFirst();
                        boom = true;
                        break;
                    } else if (deque.peekFirst() < -num) {
                        deque.pollFirst();
                    } else if (deque.peekFirst() > -num) {
                        boom = true;
                        break;
                    }
                }
                if (!boom) {
                    deque.push(num);
                }
            } else {
                deque.push(num);
            }
        }
        int[] res = new int[deque.size()];
        int n = deque.size();
        for (int i = 0; i < n; i++) {
            res[i] = deque.pollLast();
        }
        return res;
    }

    /**
     * 给定一个经过编码的字符串，返回它解码后的字符串。
     * 编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
     * 你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
     * 此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
     * 输入：s = "3[a]2[bc]"
     * 输出："aaabcbc"
     * 输入：s = "3[a2[c]]"
     * 输出："accaccacc"
     * @param s
     * @return
     */
    public String decodeString(String s) {
        //存储[]前面的字符
        Stack<String> stack = new Stack<>();
        //存储次数
        Stack<Integer> k = new Stack<>();
        //存储[]里面的字符
        StringBuilder res = new StringBuilder();
        int m = 0;
        for (char c : s.toCharArray()) {
            if ('[' == c) {
                //遇到[先把[前面的字符放到栈
                stack.push(res.toString());
                k.push(m);
                m = 0;
                res = new StringBuilder();
            } else if (']' == c) {
                int tmpK = k.pop();
                //遇到]从res里面获取[]的数据
                StringBuilder tmp = new StringBuilder();
                for (int i = 0; i < tmpK; i++) {
                    tmp.append(res);
                }
                //res再一次变成外面那层[]的数据
                res = new StringBuilder(stack.pop() + tmp);
            } else if (c >= '0' && c <= '9') {
                m = 10 * m + Integer.parseInt(c + "");
            } else {
                res.append(c);
            }
        }
        return res.toString();
    }

    /**
     * 写一个 RecentCounter 类来计算特定时间范围内最近的请求。
     * 请你实现 RecentCounter 类：
     * RecentCounter() 初始化计数器，请求数为 0 。
     * int ping(int t) 在时间 t 添加一个新请求，其中 t 表示以毫秒为单位的某个时间，
     * 并返回过去 3000 毫秒内发生的所有请求数（包括新请求）。确切地说，返回在 [t-3000, t] 内发生的请求数。
     * 保证 每次对 ping 的调用都使用比之前更大的 t 值。
     * 输入：
     * ["RecentCounter", "ping", "ping", "ping", "ping"]
     * [[], [1], [100], [3001], [3002]]
     * 输出：
     * [null, 1, 2, 3, 3]
     *
     * 解释：
     * RecentCounter recentCounter = new RecentCounter();
     * recentCounter.ping(1);     // requests = [1]，范围是 [-2999,1]，返回 1
     * recentCounter.ping(100);   // requests = [1, 100]，范围是 [-2900,100]，返回 2
     * recentCounter.ping(3001);  // requests = [1, 100, 3001]，范围是 [1,3001]，返回 3
     * recentCounter.ping(3002);  // requests = [1, 100, 3001, 3002]，范围是 [2,3002]，返回 3
     */
    class RecentCounter {

        private Queue<Integer> queue;
        public RecentCounter() {
            queue = new ArrayDeque<>();
        }

        public int ping(int t) {
            queue.add(t);
            while (!queue.isEmpty() && queue.peek() < t - 3000) {
                queue.poll();
            }
            return queue.size();
        }
    }

    /**
     * Dota2 的世界里有两个阵营：Radiant（天辉）和 Dire（夜魇）
     * Dota2 参议院由来自两派的参议员组成。现在参议院希望对一个 Dota2 游戏里的改变作出决定。
     * 他们以一个基于轮为过程的投票进行。在每一轮中，每一位参议员都可以行使两项权利中的 一 项：
     * 禁止一名参议员的权利：参议员可以让另一位参议员在这一轮和随后的几轮中丧失 所有的权利 。
     * 宣布胜利：如果参议员发现有权利投票的参议员都是 同一个阵营的 ，他可以宣布胜利并决定在游戏中的有关变化。
     * 给你一个字符串 senate 代表每个参议员的阵营。字母 'R' 和 'D'分别代表了 Radiant（天辉）和 Dire（夜魇）。
     * 然后，如果有 n 个参议员，给定字符串的大小将是 n。
     * 以轮为基础的过程从给定顺序的第一个参议员开始到最后一个参议员结束。这一过程将持续到投票结束。所有失去权利的参议员将在过程中被跳过。
     * 假设每一位参议员都足够聪明，会为自己的政党做出最好的策略，你需要预测哪一方最终会宣布胜利并在 Dota2 游戏中决定改变。
     * 输出应该是 "Radiant" 或 "Dire" 。
     * 输入：senate = "RD"
     * 输出："Radiant"
     * 解释：
     * 第 1 轮时，第一个参议员来自 Radiant 阵营，他可以使用第一项权利让第二个参议员失去所有权利。
     * 这一轮中，第二个参议员将会被跳过，因为他的权利被禁止了。
     * 第 2 轮时，第一个参议员可以宣布胜利，因为他是唯一一个有投票权的人。
     * @param senate
     * @return
     */
    public String predictPartyVictory(String senate) {
        //记录天辉和夜魇参议员的下标
        Queue<Integer> radiant = new ArrayDeque<>();
        Queue<Integer> dire = new ArrayDeque<>();
        int n = senate.length();
        for (int i = 0; i < n; i++) {
            if (senate.charAt(i) == 'R') {
                radiant.offer(i);
            } else {
                dire.offer(i);
            }
        }
        while (!radiant.isEmpty() && !dire.isEmpty()) {
            int radiantNum = radiant.poll();
            int direNum = dire.poll();
            //下标越小越靠前，越有权利禁用下标大的敌方，
            // 然后长度 + n进入下一轮，如果小于n可能会出现同一轮禁用两次的情况
            if (radiantNum < direNum) {
                radiant.offer(radiantNum + n);
            } else {
                dire.offer(direNum + n);
            }
        }
        return !radiant.isEmpty() ? "Radiant" : "Dire";
    }

    /**
     * 给你一个链表的头节点 head 。删除 链表的 中间节点 ，并返回修改后的链表的头节点 head 。
     * 长度为 n 链表的中间节点是从头数起第 ⌊n / 2⌋ 个节点（下标从 0 开始），其中 ⌊x⌋ 表示小于或等于 x 的最大整数。
     * 对于 n = 1、2、3、4 和 5 的情况，中间节点的下标分别是 0、1、1、2 和 2 。
     * 输入：head = [1,3,4,7,1,2,6]
     * 输出：[1,3,4,1,2,6]
     * 解释：
     * 上图表示给出的链表。节点的下标分别标注在每个节点的下方。
     * 由于 n = 7 ，值为 7 的节点 3 是中间节点，用红色标注。
     * 返回结果为移除节点后的新链表。
     * @param head
     * @return
     */
    public ListNode deleteMiddle(ListNode head) {
        //快慢指针，再使用一个哨兵节点
        ListNode pre = new ListNode();
        pre.next = head;
        ListNode slow = pre;
        ListNode fast = pre.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        slow.next = slow.next.next;
        return pre.next;
    }

    /**
     * 给定单链表的头节点 head ，将所有索引为奇数的节点和索引为偶数的节点分别组合在一起，然后返回重新排序的列表。
     * 第一个节点的索引被认为是 奇数 ， 第二个节点的索引为 偶数 ，以此类推。
     * 请注意，偶数组和奇数组内部的相对顺序应该与输入时保持一致。
     * 你必须在 O(1) 的额外空间复杂度和 O(n) 的时间复杂度下解决这个问题。
     * 输入: head = [1,2,3,4,5]
     * 输出: [1,3,5,2,4]
     * @param head
     * @return
     */
    public ListNode oddEvenList(ListNode head) {
        if (head == null) {
            return null;
        }
        //奇数指针
        ListNode slow = head;
        //偶数指针
        ListNode fast = head.next;
        while (fast != null && fast.next != null) {
            ListNode tmp = slow.next;
            slow.next = fast.next;
            fast.next = fast.next.next;
            slow = slow.next;
            slow.next = tmp;
            fast = fast.next;
        }
        return head;
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
     * 在一个大小为 n 且 n 为 偶数 的链表中，对于 0 <= i <= (n / 2) - 1 的 i ，第 i 个节点（下标从 0 开始）的孪生节点为第 (n-1-i) 个节点 。
     * 比方说，n = 4 那么节点 0 是节点 3 的孪生节点，节点 1 是节点 2 的孪生节点。这是长度为 n = 4 的链表中所有的孪生节点。
     * 孪生和 定义为一个节点和它孪生节点两者值之和。
     * 给你一个长度为偶数的链表的头节点 head ，请你返回链表的 最大孪生和 。
     * 输入：head = [5,4,2,1]
     * 输出：6
     * 解释：
     * 节点 0 和节点 1 分别是节点 3 和 2 的孪生节点。孪生和都为 6 。
     * 链表中没有其他孪生节点。
     * 所以，链表的最大孪生和是 6 。
     * @param head
     * @return
     */
    public int pairSum(ListNode head) {
        int n = 0;
        ListNode pre = new ListNode(0, head);
        ListNode slow = pre;
        ListNode fast = pre.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            n++;
        }
        int[] sum = new int[n];
        fast = slow.next;
        slow = head;
        for (int i = 0; i < n; i++) {
            sum[i] += slow.val;
            sum[n - i - 1] += fast.val;
            slow = slow.next;
            fast = fast.next;
        }
        int res = sum[0];
        for (int i = 1; i < n ;i++) {
            if (sum[i] > res) {
                res = sum[i];
            }
        }
        return res;
    }

    /**
     * 给定一个二叉树，找出其最大深度。
     * 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
     * 说明: 叶子节点是指没有子节点的节点。
     * 示例：
     * 给定二叉树 [3,9,20,null,null,15,7]，
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }

    /**
     * 请考虑一棵二叉树上所有的叶子，这些叶子的值按从左到右的顺序排列形成一个 叶值序列 。
     * 举个例子，如上图所示，给定一棵叶值序列为 (6, 7, 4, 9, 8) 的树。
     * 如果有两棵二叉树的叶值序列是相同，那么我们就认为它们是 叶相似 的。
     * 如果给定的两个根结点分别为 root1 和 root2 的树是叶相似的，则返回 true；否则返回 false 。
     * @param root1
     * @param root2
     * @return
     */
    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        leafSimilar_help(root1, list1);
        leafSimilar_help(root2, list2);
        if (list1.size() != list2.size()) {
            return false;
        }
        for (int i = 0; i < list1.size(); i++) {
            if (!list1.get(i).equals(list2.get(i))) {
                return false;
            }
        }
        return true;
    }

    private void leafSimilar_help(TreeNode node, List<Integer> list) {
        if (node == null) {
            return;
        }
        if (node.left == null && node.right == null) {
            list.add(node.val);
            return;
        }
        leafSimilar_help(node.left, list);
        leafSimilar_help(node.right, list);
    }

    /**
     * 给你一棵根为 root 的二叉树，请你返回二叉树中好节点的数目。
     * 「好节点」X 定义为：从根到该节点 X 所经过的节点中，没有任何节点的值大于 X 的值。
     * 输入：root = [3,1,4,3,null,1,5]
     * 输出：4
     * 解释：图中蓝色节点为好节点。
     * 根节点 (3) 永远是个好节点。
     * 节点 4 -> (3,4) 是路径中的最大值。
     * 节点 5 -> (3,4,5) 是路径中的最大值。
     * 节点 3 -> (3,1,3) 是路径中的最大值。
     * @param root
     * @return
     */
    int goodNodes_res = 0;
    public int goodNodes(TreeNode root) {
        goodNodes_help(root, Integer.MIN_VALUE);
        return goodNodes_res;
    }

    private void goodNodes_help(TreeNode root, int max) {
        if (root == null) {
            return;
        }
        if (root.val >= max) {
            goodNodes_res++;
            max = root.val;
        }
        goodNodes_help(root.left, max);
        goodNodes_help(root.right, max);
    }

    /**
     * 给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
     * 路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
     * 输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
     * 输出：3
     * 解释：和等于 8 的路径有 3 条，如图所示。
     * @param root
     * @param targetSum
     * @return
     */
    public int pathSum(TreeNode root, int targetSum) {
        Map<Long, Integer> prefix = new HashMap<>();
        prefix.put(0L, 1);
        return pathSum_help(root, targetSum, 0L, prefix);
    }

    private int pathSum_help(TreeNode root, int targetSum, Long cur, Map<Long, Integer> prefix) {
        if (root == null) {
            return 0;
        }
        cur += root.val;
        //前缀和 - targetSum如果出现在Map中，证明路径上可以组合出等于target
        int res = prefix.getOrDefault(cur - targetSum, 0);
        //可能出现相同的前缀和
        prefix.put(cur, prefix.getOrDefault(cur, 0) + 1);
        res += pathSum_help(root.left, targetSum, cur, prefix);
        res += pathSum_help(root.right, targetSum, cur, prefix);
        prefix.put(cur, prefix.get(cur) - 1);
        return res;
    }

    /**
     * 给你一棵以 root 为根的二叉树，二叉树中的交错路径定义如下：
     * 选择二叉树中 任意 节点和一个方向（左或者右）。
     * 如果前进方向为右，那么移动到当前节点的的右子节点，否则移动到它的左子节点。
     * 改变前进方向：左变右或者右变左。
     * 重复第二步和第三步，直到你在树中无法继续移动。
     * 交错路径的长度定义为：访问过的节点数目 - 1（单个节点的路径长度为 0 ）。
     * 请你返回给定树中最长 交错路径 的长度。
     * 输入：root = [1,null,1,1,1,null,null,1,1,null,1,null,null,null,1,null,1]
     * 输出：3
     * 解释：蓝色节点为树中最长交错路径（右 -> 左 -> 右）。
     * @param root
     * @return
     */
    int longestZigZag_res = 0;
    public int longestZigZag(TreeNode root) {
        if (root == null) {
            return 0;
        }
        longestZigZag_help(root, 0, true);
        longestZigZag_help(root, 0, false);
        return longestZigZag_res;
    }

    private void longestZigZag_help(TreeNode root, int sum, boolean dir) {
        longestZigZag_res = Math.max(longestZigZag_res, sum);
        //true为要向左走，false为要向右走
        if (dir) {
            if (root.left != null) {
                longestZigZag_help(root.left, sum + 1, false);
            }
            if (root.right != null) {
                //true为要向左走，如果走右边则重置sum为1
                longestZigZag_help(root.right, 1, true);
            }
        } else {
            if (root.left != null) {
                longestZigZag_help(root.left, 1, false);
            }
            if (root.right != null) {
                longestZigZag_help(root.right, sum + 1, true);
            }
        }
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
     * 给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
     * @param root
     * @return
     */
    public List<Integer> rightSideView(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> res = new ArrayList<>();
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            res.add(queue.peek().val);
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
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
     * 给你一个二叉树的根节点 root。设根节点位于二叉树的第 1 层，而根节点的子节点位于第 2 层，依此类推。
     * 请返回层内元素之和 最大 的那几层（可能只有一层）的层号，并返回其中 最小 的那个。
     * 输入：root = [1,7,0,7,-8,null,null]
     * 输出：2
     * 解释：
     * 第 1 层各元素之和为 1，
     * 第 2 层各元素之和为 7 + 0 = 7，
     * 第 3 层各元素之和为 7 + -8 = -1，
     * 所以我们返回第 2 层的层号，它的层内元素之和最大。
     * @param root
     * @return
     */
    public int maxLevelSum(TreeNode root) {
        int res = 0;
        int max = Integer.MIN_VALUE;
        int dep = 0;
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            dep++;
            int size = queue.size();
            int sum = 0;
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                sum += node.val;
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            if (sum > max) {
                res = dep;
                max = sum;
            }
        }
        return res;
    }

    /**
     * 给定二叉搜索树（BST）的根节点 root 和一个整数值 val。
     * 你需要在 BST 中找到节点值等于 val 的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 null 。
     * @param root
     * @param val
     * @return
     */
    public TreeNode searchBST(TreeNode root, int val) {
        while (root != null) {
            if (root.val == val) {
                return root;
            } else if (root.val < val) {
                root = root.right;
            } else if (root.val > val) {
                root = root.left;
            }
        }
        return null;
    }

    /**
     * 给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，
     * 并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。
     * 一般来说，删除节点可分为两个步骤：
     * 首先找到需要删除的节点；
     * 如果找到了，删除它。
     * @param root
     * @param key
     * @return
     */
    public TreeNode deleteNode(TreeNode root, int key) {
        //TODO
        return null;
    }

    /**
     * 有 n 个房间，房间按从 0 到 n - 1 编号。最初，除 0 号房间外的其余所有房间都被锁住。
     * 你的目标是进入所有的房间。然而，你不能在没有获得钥匙的时候进入锁住的房间。
     * 当你进入一个房间，你可能会在里面找到一套不同的钥匙，每把钥匙上都有对应的房间号，即表示钥匙可以打开的房间。你可以拿上所有钥匙去解锁其他房间。
     * 给你一个数组 rooms 其中 rooms[i] 是你进入 i 号房间可以获得的钥匙集合。如果能进入 所有 房间返回 true，否则返回 false。
     * 输入：rooms = [[1],[2],[3],[]]
     * 输出：true
     * 解释：
     * 我们从 0 号房间开始，拿到钥匙 1。
     * 之后我们去 1 号房间，拿到钥匙 2。
     * 然后我们去 2 号房间，拿到钥匙 3。
     * 最后我们去了 3 号房间。
     * 由于我们能够进入每个房间，我们返回 true。
     * [[6,7,8],[5,4,9],[],[8],[4],[],[1,9,2,3],[7],[6,5],[2,3,1]]
     * @param rooms
     * @return
     */
    boolean[] canVisitAllRooms_dp;
    int canVisitAllRooms_res = 0;
    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        int n = rooms.size();
        canVisitAllRooms_dp = new boolean[n];
        canVisitAllRooms_help(rooms, 0);
        return canVisitAllRooms_res == n;
    }

    private void canVisitAllRooms_help(List<List<Integer>> rooms, int index) {
        canVisitAllRooms_dp[index] = true;
        canVisitAllRooms_res++;
        for (int i : rooms.get(index)) {
            if (!canVisitAllRooms_dp[i]) {
                canVisitAllRooms_help(rooms, i);
            }
        }
    }

    /**
     * 有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。
     * 省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。
     * 给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，
     * 而 isConnected[i][j] = 0 表示二者不直接相连。
     * 返回矩阵中 省份 的数量。
     * 输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]]
     * 输出：2
     * @param isConnected
     * @return
     */
    public int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        boolean[] visited = new boolean[n];
        int res = 0;
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                findCircleNum_help(isConnected, visited, n, i);
                res++;
            }
        }
        return res;
    }

    private void findCircleNum_help(int[][] isConnected, boolean[] visited, int n , int i) {
        for (int j = 0; j < n; j++) {
            if (isConnected[i][j] == 1 && !visited[j]) {
                visited[j] = true;
                findCircleNum_help(isConnected, visited, n, j);
            }
        }
    }

    /**
     * 给你一个 m x n 的迷宫矩阵 maze （下标从 0 开始），矩阵中有空格子（用 '.' 表示）和墙（用 '+' 表示）。
     * 同时给你迷宫的入口 entrance ，用 entrance = [entrancerow, entrancecol] 表示你一开始所在格子的行和列。
     * 每一步操作，你可以往 上，下，左 或者 右 移动一个格子。你不能进入墙所在的格子，你也不能离开迷宫。
     * 你的目标是找到离 entrance 最近 的出口。出口 的含义是 maze 边界 上的 空格子。entrance 格子 不算 出口。
     * 请你返回从 entrance 到最近出口的最短路径的 步数 ，如果不存在这样的路径，请你返回 -1 。
     * 输入：maze = [["+","+",".","+"],[".",".",".","+"],["+","+","+","."]], entrance = [1,2]
     * 输出：1
     * 解释：总共有 3 个出口，分别位于 (1,0)，(0,2) 和 (2,3) 。
     * 一开始，你在入口格子 (1,2) 处。
     * - 你可以往左移动 2 步到达 (1,0) 。
     * - 你可以往上移动 1 步到达 (0,2) 。
     * 从入口处没法到达 (2,3) 。
     * 所以，最近的出口是 (0,2) ，距离为 1 步。
     * @param maze
     * @param entrance
     * @return
     */
    public int nearestExit(char[][] maze, int[] entrance) {
        int m = maze.length;
        int n = maze[0].length;
        //存储上下左右移动
        int[][] move = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        //广度优先遍历使用队列
        Queue<int[]> queue = new ArrayDeque<>();
        queue.add(entrance);
        boolean[][] visited = new boolean[m][n];
        visited[entrance[0]][entrance[1]] = true;
        int res = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int k = 0; k < size; k++) {
                int[] tmp = queue.poll();
                for (int i = 0; i < 4; i++) {
                    int x = tmp[0] + move[i][0];
                    int y = tmp[1] + move[i][1];
                    //如果x或者y越界，说明tmp的下标已经到了边缘
                    if (x < 0 || x >= m || y < 0 || y >= n) {
                        //只要tmp中的下标任意一个不等于起点就是到了出口
                        if (entrance[0] != tmp[0] || entrance[1] != tmp[1]) {
                            return res;
                        }
                    } else if (!visited[x][y] && maze[x][y] == '.') {
                        //未访问过的进入队列
                        queue.offer(new int[]{x, y});
                        visited[x][y] = true;
                    }
                }
            }
            res++;
        }
        return -1;
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
        int[][] move = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        int res = 0;
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
            for (int k = 0; k < size; k++) {
                int[] tmp = queue.poll();
                for (int i = 0; i < 4; i++) {
                    int x = tmp[0] + move[i][0];
                    int y = tmp[1] + move[i][1];
                    if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == 1) {
                        queue.offer(new int[]{x, y});
                        grid[x][y] = 2;
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

    public static void main(String[] args) {
        LeetCode75 leetCode75 = new LeetCode75();
        leetCode75.orangesRotting(new int[][]{{2, 1, 1}, {1, 1, 0}, {0, 1, 1}});
    }
}
