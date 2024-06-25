import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Solution {

    private class Node {
        String parent;
        String owner;
        int rank;

        public Node(String owner, String parent, int rank) {
            this.owner = owner;
            this.parent = parent;
            this.rank = rank;
        }

        List<List<Integer>> permuteResult;

        // problem link - https://leetcode.com/problems/01-matrix/description/

        public int[][] updateMatrix(int[][] mat) {
            int m = mat.length;
            int n = mat[0].length;
            int[][] result = new int[m][n];
            for (int[] a : result) {
                Arrays.fill(a, -1);
            }
            Queue<int[]> queue = new LinkedList<>();
            int[][] directions = new int[][]{{0, 1}, {0, -1}, {1, 0}, {0, -1}};
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (mat[i][j] == 0) {
                        result[i][j] = 0;
                        queue.add(new int[]{i, j});
                    }
                }
            }
            while (!queue.isEmpty()) {
                int[] p = queue.remove();
                int i = p[0];
                int j = p[1];

                for (int[] d : directions) {
                    int new_i = i + d[0];
                    int new_j = j + d[1];

                    if (new_i >= 0 && new_i < m && new_j >= 0 && new_j < n && result[new_i][new_j] == -1) {
                        result[new_i][new_j] = 1 + result[i][j];
                        queue.add(new int[]{new_i, new_j});
                    }
                }
            }
            return result;
        }

        // problem link - https://leetcode.com/problems/k-closest-points-to-origin/description/
        public int[][] kClosest(int[][] points, int k) {
            PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>(){
                public int compare(int[] n1,int[] n2) {
                    // compare n1 and n2
                    if((Math.pow(n1[0],2)+Math.pow(n1[1],2))>(Math.pow(n2[0],2)+Math.pow(n2[1],2))) return 1;
                    else return -1;
                }
            });

            for(int i=0;i<points.length;i++){
                pq.offer(points[i]);
            }

            int i=0;
            int ans[][] = new int[k][2];
            while(k-->0){
                int arr[] = pq.poll();

                ans[i++] = arr;

            }
            return ans;
        }

        public boolean canFinish(int numCourses, int[][] prerequisites) {
            Map<Integer, List<Integer>> adj = new HashMap<>();
            for (int i = 0; i < numCourses; i++) {
                adj.put(i, new ArrayList<>());
            }
            for (int[] pre : prerequisites) {
                if (adj.containsKey(pre[0])) {
                    List<Integer> list = adj.get(pre[0]);
                    list.add(pre[1]);
                    adj.put(pre[0], list);
                }
            }
            Set<Integer> visited = new HashSet<>();
            for (int i = 0; i < numCourses; i++) {
                if (!dfs(i, adj, visited)) {
                    return false;
                }
            }
            return true;

        }

        private boolean dfs(int course, Map<Integer, List<Integer>> adj, Set<Integer> visited) {
            if (visited.contains(course)) {
                return false;
            }
            if (adj.get(course).isEmpty()) {
                return true;
            }
            visited.add(course);
            for (int pre : adj.get(course)) {
                if (!dfs(pre, adj, visited)) {
                    return false;
                }
            }
            visited.remove(course);
            adj.get(course).clear();
            return true;
        }

        public int[] productExceptSelf(int[] nums) {
            int[] output = new int[nums.length];
            output[0] = 1;
            for (int i = 1; i < nums.length; i++) {
                output[i] = output[i - 1] * nums[i - 1];
            }
            int r = 1;
            for (int j = nums.length - 1; j >= 0; j++) {
                output[j] = output[j] * r;
                r = nums[j] * r;
            }
            return output;
        }

        // https://leetcode.com/problems/validate-binary-search-tree/
        public boolean isValidBST(TreeNode root) {
            return backtrack(root, null, null);
        }
        private boolean backtrack(TreeNode root, Integer low, Integer high){
            if(root == null) return true;
            if(low != null && root.val <= low || high != null && root.val >= high) return false;
            return backtrack(root.left,low,root.val) && backtrack(root.right,root.val,high);
        }

        public int numIslands(char[][] grid) {
            int counter = 0;
            int m = grid.length;
            int n = grid[0].length;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (grid[i][j] == 1) {
                        counter++;
                        dfsGrid(grid, i, j);
                    }
                }
            }
            return counter;
        }

        private void dfsGrid(char[][] grid, int i, int j) {
            if (!isSafe(grid, i, j)) {
                return;
            }
            grid[i][j] = 0;
            dfsGrid(grid, i + 1, j);
            dfsGrid(grid, i - 1, j);
            dfsGrid(grid, i, j + 1);
            dfsGrid(grid, i, j - 1);
        }

        private boolean isSafe(char[][] grid, int i, int j) {
            if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] == 0) {
                return false;
            }
            return true;
        }

        public int orangesRotting(int[][] grid) {
            Queue<int[]> queue = new LinkedList<>();
            int[][] directions = new int[][]{{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
            int time = 0, fresh = -1;
            int rows = grid.length;
            int cols = grid[0].length;

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    if (grid[r][c] == 1) {
                        fresh++;
                    }
                    if (grid[r][c] == 2) {
                        queue.add(new int[]{r, c});
                    }
                }
            }

            while (!queue.isEmpty() && fresh > 0) {
                for (int i = 0; i < queue.size(); i++) {
                    int[] curr = queue.poll();
                    for (int[] direction : directions) {
                        int R = direction[0] + curr[0];
                        int C = direction[1] + curr[1];

                        if (R < 0 || R == rows
                                || C < 0 || C == cols
                                || grid[R][C] != 1) {
                            continue;
                        }
                        grid[R][C] = 2;
                        queue.add(new int[]{R, C});
                        fresh--;
                    }
                }
                time++;
            }
            return fresh == 0 ? time : -1;
        }

        public List<List<Integer>> subsets(int[] nums) {
            List<List<Integer>> list = new ArrayList<>();
            Arrays.sort(nums);
            backtrack(list, new ArrayList<>(), nums, 0);
            return list;
        }

        private void backtrack(List<List<Integer>> list, List<Integer> tempList, int[] nums, int start) {
            list.add(new ArrayList<>(tempList));
            for (int i = start; i < nums.length; i++) {
                tempList.add(nums[i]);
                backtrack(list, tempList, nums, i + 1);
                tempList.remove(tempList.size() - 1);
            }
        }

        public List<List<Integer>> combinationSum(int[] nums, int target) {
            List<List<Integer>> list = new ArrayList<>();
            Arrays.sort(nums);
            backtrack(list, new ArrayList<>(), nums, target, 0);
            return list;
        }

        private void backtrack(List<List<Integer>> list, List<Integer> tempList, int[] nums, int remain, int start) {
            if (remain < 0) return;
            else if (remain == 0) list.add(new ArrayList<>(tempList));
            else {
                for (int i = start; i < nums.length; i++) {
                    tempList.add(nums[i]);
                    backtrack(list, tempList, nums, remain - nums[i], i); // not i + 1 because we can reuse same elements
                    tempList.remove(tempList.size() - 1);
                }
            }
        }

        public List<List<Integer>> permute(int[] nums) {
            permuteResult = new ArrayList<>();
            if (nums.length == 0) {
                return permuteResult;
            }
            solvePermute(nums, 0);
            return permuteResult;
        }

        private void solvePermute(int[] nums, int index) {
            if (index >= nums.length) {
                List<Integer> temp = IntStream.of(nums).boxed().collect(Collectors.toList());
                permuteResult.add(temp);
                return;
            }
            for (int i = 0; i < index; i++) {
                swap(nums, index, i);
                solvePermute(nums, index + 1);
                swap(nums, index, i);
            }
        }

        private void swap(int[] nums, int index, int i) {
            int temp = nums[index];
            nums[index] = nums[i];
            nums[i] = temp;
        }

        // https://leetcode.com/problems/merge-intervals/
        public int[][] merge(int[][] intervals) {
            List<int[]> ans = new ArrayList<>();

            Arrays.sort(intervals, Comparator.comparing(value -> value[0]));

            for(int[] interval: intervals) {
                if(ans.isEmpty() || ans.get(ans.size()-1)[1] < interval[0]) {
                    ans.add(interval);
                } else {
                    ans.get(ans.size()-1)[1] = Math.max(ans.get(ans.size()-1)[1], interval[1]);
                }
            }

            return ans.toArray(new int[ans.size()][]);
        }

        public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
            if (root == null) {
                return null;
            }
            if (root == p || root == q) {
                return root;
            }
            TreeNode left = lowestCommonAncestor(root.left, p, q);
            TreeNode right = lowestCommonAncestor(root.right, p, q);

            if (left != null && right != null) {
                return root;
            }

            return left != null ? left : right;
        }

    }
    public void sortColors(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        int i = 0;

        while(i <= right) {
            if(nums[i] == 0) {
                swap(nums, i, left);
                left++;
                i++;
            } else if(nums[i] == 2) {
                swap(nums, i, right);
                right--;
            } else {
                i++;
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        boolean[] arr = new boolean[n+1];
        return solveWordBreak(s, wordDict, n, arr, 0);
    }

    private boolean solveWordBreak(String s, List<String> wordDict, int n, boolean[] arr, int start) {
        if(start == n) {
            return true;
        }
        for(String word: wordDict) {
            if(s.startsWith(word, start)) {
                int len = start + word.length();
                if(!arr[len]) {
                    if (solveWordBreak(s, wordDict, n, arr, len)) return true;
                    arr[len] = true;
                }
            }
        }
        return false;
    }

    public int myAtoi(String s) {
        int len = s.length();
        if(len == 0) {
            return 0;
        }
        int index = 0;
        while(index < len && s.charAt(index) == ' ') {
            index++;
        }
        boolean isNegative = false;
        if(index < len) {
            if(s.charAt(index) == '-') {
                isNegative = true;
                ++index;
            } else if(s.charAt(index) == '+') {
                ++index;
            }
        }
        int result = 0;
        while (index < len && isDigit(s.charAt(index))) {
            int digit = s.charAt(index) - '0';
            if (result > (Integer.MAX_VALUE / 10) || (result == (Integer.MAX_VALUE / 10) && digit > 7)){
                return isNegative ? Integer.MIN_VALUE : Integer.MAX_VALUE;
            }
            result = (result * 10) + digit;
            ++index;
        }
        return isNegative ? -result : result;
    }

    private boolean isDigit(char ch) {
        return ch >= '0' && ch <= '9';
    }

    Boolean mem[][];
    public boolean canPartition(int[] nums) {
        int sum = 0;
        int n = nums.length;

        for(int i : nums) sum+=i;

        if(sum%2!=0) return false;

        sum /= 2;

        mem = new Boolean[n+1][sum+1];

        return subsetSum(nums,0,sum);
    }

    boolean subsetSum(int[] nums, int pos, int sum){
        if(sum==0) return true;

        else if(pos>=nums.length || sum<0) return false;

        if(mem[pos][sum]!=null) return mem[pos][sum];

        return mem[pos][sum] = subsetSum(nums,pos+1,sum-nums[pos]) ||
                subsetSum(nums,pos+1,sum);


    }
    public List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int left = 0, right = n-1, top = 0, bottom = m-1;
        List<Integer> result = new ArrayList<>();

        while(top <= bottom && left <= right) {
            //print left to right
            for(int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
            }
            top++;
            // print top to bottom
            for(int i = top; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }
            right--;
            if(top <= bottom) {
                //print right to left
                for(int i = right; i >= left; i--) {
                    result.add(matrix[bottom][i]);
                }
                bottom--;
            }

            if(left <= right) {
                //print bottom to top
                for(int i = bottom; i >= top; i--) {
                    result.add(matrix[i][left]);
                }
                left++;
            }
        }

        return result;
    }

    List<List<Integer>> powerSet;
    public List<List<Integer>> subsets(int[] nums) {
        powerSet = new ArrayList<>();
        solveSubsets(nums, 0, new ArrayList<>());
        return powerSet;
    }

    private void solveSubsets(int[] nums, int index, List<Integer> output) {
        if(index == nums.length) {
            powerSet.add(new ArrayList<>(output));
            return;
        }
        solveSubsets(nums, index+1, output);
        output.add(nums[index]);
        solveSubsets(nums, index+1, output);
        output.remove(output.size()-1);
    }

    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if(root == null) {
            return null;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()) {
            int level = queue.size();
            for (int i = 0; i < level; i++) {
                TreeNode cuurentNode = queue.poll();
                if(i == level - 1) {
                    result.add(cuurentNode.val);
                }
                if(cuurentNode.left != null) {
                    queue.offer(cuurentNode.left);
                }
                if(cuurentNode.right != null) {
                    queue.offer(cuurentNode.right);
                }
            }
        }
        return result;
    }

    public String longestPalindrome(String s) {
        if(s.length() <= 1) {
            return s;
        }
        String maxStr = s.substring(0, 1);

        for(int i = 0; i < s.length() - 1; i++) {
            String odd = expandFromCenter(s, i, i);
            String even = expandFromCenter(s, i, i+1);

            if (odd.length() > maxStr.length()) {
                maxStr = odd;
            }
            if (even.length() > maxStr.length()) {
                maxStr = even;
            }
        }

        return maxStr;
    }

    private String expandFromCenter(String s, int left, int right) {
        while( left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return s.substring(left+1, right);
    }

    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for(int[] arr: dp) {
            Arrays.fill(arr, -1);
        }
        return solveUniquePaths(0, 0, m, n, dp);
    }

    private int solveUniquePaths(int x, int y, int m, int n, int[][] dp) {
        if(x == n-1 && y == m-1) {
            return 1;
        }
        if(x >= n || y >= m) {
            return 0;
        } else {
            if(dp[y][x] != -1) {
                return dp[y][x];
            } else
                return dp[y][x] = solveUniquePaths(x+1, y, m, n, dp) + solveUniquePaths(x, y+1, m, n, dp);
        }

    }

    Map<Integer, Integer> nodeToindex;
    int preOrderIndex = 0;
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        nodeToindex = new HashMap<>();
        for(int i=0; i < inorder.length; i++) {
            nodeToindex.put(inorder[i], i);
        }
        return solveBuildTree(preorder, inorder, 0, inorder.length-1, preorder.length);
    }

    private TreeNode solveBuildTree(int[] preorder, int[] inorder, int inOrderStart, int inorderEnd, int n) {
        if(preOrderIndex >= n || inOrderStart > inorderEnd) {
            return null;
        }
        int element = preorder[preOrderIndex++];
        TreeNode root = new TreeNode(element);
        int position = nodeToindex.get(element);

        root.left = solveBuildTree(preorder, inorder, inOrderStart, position-1, n);
        root.right = solveBuildTree(preorder, inorder, position+1, inorderEnd, n);

        return root;
    }

    public int maxArea(int[] height) {
        int maxArea = 0;
        int left = 0;
        int right = height.length - 1;
        while(left <= right) {
            maxArea = Math.max(maxArea, (right - left) * Math.min(height[left], height[right]));
            if(height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return maxArea;
    }

    List<String> result = new ArrayList<>();
    String digits;
    Map<Character, String> digitToChar;
    public List<String> letterCombinations(String digits) {
        this.digits = digits;
        digitToChar = Map.of(
                '2', "abc",
                '3', "def",
                '4', "ghi",
                '5', "jkl",
                '6', "mno",
                '7', "pqrs",
                '8', "tuv",
                '9', "wzyz"
        );
        if(digits.length() > 0) {
            solveLetterCombinations(0, "");
        }
        return result;
    }

    private void solveLetterCombinations(int index, String current) {
        if(current.length() == digits.length()) {
            result.add(current);
            return;
        }
        for(char c: digitToChar.get(digits.charAt(index)).toCharArray()) {
            solveLetterCombinations(index+1, current + c);
        }
    }

    public boolean exist(char[][] board, String word) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if(find(board,word,i,j,0)) // check for every char
                    return true;
            }
        }
        return false;
    }
    private boolean find(char[][] board, String word, int i, int j, int index) {
        if(index==word.length()) // word found
            return true;
        if(i<0 || j<0 || i>=board.length || j>=board[0].length || board[i][j]!=word.charAt(index)) //outofbounds and char not matching exception
            return false;
        char temp=board[i][j];
        board[i][j]=' ';

        //check in all 4 directions
        boolean found = find(board,word,i+1,j,index+1) ||
                find(board,word,i,j+1,index+1) ||
                find(board,word,i-1,j,index+1) ||
                find(board,word,i,j-1,index+1);

        board[i][j]=temp;
        return found;
    }

    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> result = new ArrayList<>();
        if(p.length() > s.length()) {
            return result;
        }

        HashMap<Character,Integer> map=new HashMap<>();
        for(int i=0; i<p.length(); i++) {
            map.put(p.charAt(i),map.getOrDefault(p.charAt(i),0)+1);
        }

        Map<Character,Integer> check=new HashMap<>();

        for(int i=0; i<p.length(); i++) {
            check.put(s.charAt(i),check.getOrDefault(s.charAt(i),0)+1);
        }

        if(map.equals(check)) {
            result.add(0);
        }

        for(int i = 0; i < s.length() - p.length(); i++) {
            int val = check.get(s.charAt(i));
            if(val == 1) {
                check.remove(s.charAt(i));
            } else {
                check.put(s.charAt(i), val-1);
            }
            check.put(s.charAt(i + p.length()), check.getOrDefault(s.charAt(i + p.length()), 0)+1);

            if(check.equals(map)) {
                result.add(i+1);
            }
        }
        return result;

    }

    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if(n == 1) {
            return Collections.singletonList(0);
        }
        List<Integer> result = new ArrayList<>();

        List<Set<Integer>> adj = new ArrayList<>();
        int[] inOrder = new int[n];

        for(int i=0; i < n; i++) {
            adj.add(new HashSet<>());
        }
        for(int[] edge: edges) {
            adj.get(edge[0]).add(edge[1]);
            adj.get(edge[1]).add(edge[0]);
            inOrder[edge[0]]++;
            inOrder[edge[1]]++;
        }

        Queue<Integer> queue = new LinkedList<>();
        for(int i = 0; i < n; i++) {
            if(inOrder[i] == 1) {
                queue.add(i);
            }
        }

        while (!queue.isEmpty()) {
            result.clear();
            for(int i=0; i < queue.size(); i++) {
                int curr = queue.poll();
                for(int adjNode: adj.get(curr)) {
                    inOrder[adjNode]--;
                    if(inOrder[adjNode] == 1) {
                        queue.add(adjNode);
                    }
                }
                result.add(curr);
            }
        }

        return result;
    }

    public int leastInterval(char[] tasks, int n) {
        Map<Character, Integer> countMap = new HashMap<>();
        for(char c: tasks) {
            countMap.put(c, countMap.getOrDefault(c, 0) + 1);
        }
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>(Collections.reverseOrder());
        for(Map.Entry<Character, Integer> entry: countMap.entrySet()){
            priorityQueue.add(entry.getValue());
        }

        int time = 0;

        Queue<int[]> queue = new LinkedList<>();

        while(!priorityQueue.isEmpty() || !queue.isEmpty()) {
            time++;
            if(!priorityQueue.isEmpty()) {
                int count = priorityQueue.poll() - 1;
                if(count > 0) {
                    queue.add(new int[]{count, time + n});
                }
            }

            if(!queue.isEmpty() && queue.peek()[1] == time) {
                priorityQueue.add(queue.poll()[0]);
            }
        }

        return time;
    }

    public int kthSmallest(TreeNode root, int k) {
        int[] ans = new int[2];
        helperKthSmallest(root, ans, k);
        return ans[1];
    }

    private void helperKthSmallest(TreeNode root, int[] nums, int k) {
        if(root == null) {
            return;
        }
        helperKthSmallest(root.left, nums, k);
        nums[0]++;
        if(nums[0] == k) {
            nums[1] = root.val;
            return;
        }
        helperKthSmallest(root.right, nums, k);
    }

    public String minWindow(String s, String t) {
        String answer = "";
        int[] tMap = new int[128];
        int[] sMap = new int[128];
        for(char c: t.toCharArray()) {
            tMap[c]++;
        }

        int matchCount = 0;
        int desiredMatchCount = t.length();

        int ptr1 = -1;
        int ptr2 = -1;

        while(true) {
            boolean f1 = false;
            boolean f2 = false;
            //acquire
            if(ptr1 < s.length() - 1 && matchCount < desiredMatchCount) {
                ptr1++;
                char ch = s.charAt(ptr1);
                sMap[ch]++;

                if(sMap[ch] <= tMap[ch]) {
                    matchCount++;
                }

                f1 = true;
            }
            //collect ans and release
            while(ptr2 < ptr1 && matchCount == desiredMatchCount) {
                String temp = s.substring(ptr2 + 1, ptr1 + 1);
                if(answer.length() == 0 || temp.length() < answer.length()) {
                    answer = temp;
                }

                ptr2++;
                char ch = s.charAt(ptr2);
                sMap[ch]--;

                if(sMap[ch] < tMap[ch]) {
                    matchCount--;
                }

                f2 = true;
            }
            if(!f1 && !f2) {
                break;
            }
        }

        return answer;
    }

    public String serialize(TreeNode root) {
        if(root == null) {
            return "";
        }

        StringBuilder result = new StringBuilder();
        Queue<TreeNode> queue = new LinkedList<>();

        queue.add(root);

        while(!queue.isEmpty()) {
            TreeNode node = queue.peek();
            if(node == null) {
                result.append("n ");
                continue;
            }
            result.append(node.val + " ");
            queue.add(node.left);
            queue.add(node.right);
        }

        return result.toString();
    }

    public TreeNode deserialize(String data) {
        if(Objects.equals(data, "")){
            return null;
        }
        Queue<TreeNode> q = new LinkedList<>();
        String values[] = data.split(" ");
        TreeNode root = new TreeNode(Integer.parseInt(values[0]));
        q.add(root);
        for(int i = 1; i < values.length; i++) {
            TreeNode parent = q.poll();
            if(!values[i].equals("n")) {
                TreeNode left = new TreeNode(Integer.parseInt(values[i]));
                parent.left = left;
                q.add(left);
            }
            if(!values[++i].equals("n")) {
                TreeNode right = new TreeNode(Integer.parseInt(values[i]));
                parent.right = right;
                q.add(right);
            }
        }
        return root;
    }

    public int trap(int[] height) {
        int ans = 0;
        int n = height.length;
        int[] prefix = new int[n];
        int[] suffix = new int[n];

        prefix[0] = height[0];
        suffix[n-1] = height[n-1];
        for(int i = 1; i < n; i++) {
            prefix[i] = Math.max(height[i], prefix[i-1]);
            int j = n - i - 1;
            suffix[j] = Math.max(height[j], suffix[j+1]);
        }

        for(int i = 0; i < n; i++) {
            ans = ans + Math.min(prefix[i], suffix[i]) - height[i];
        }
        return ans;
    }

    public int trap2(int[] height) {
        int ans = 0;
        int n = height.length;

        int l = 0, r = n - 1;
        int lMax = height[l], rMax = height[r];

        while(l < r) {
            if(lMax < rMax) {
                l++;
                lMax = Math.max(height[l], lMax);
                ans = ans + lMax - height[l];
            } else {
                r--;
                rMax = Math.max(height[r], rMax);
                ans = ans + rMax - height[r];
            }
        }

        return ans;
    }

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Queue<Pair<String, Integer>> queue = new LinkedList<>();
        queue.add(new Pair<>(beginWord, 1));
        Set <String> st = new HashSet<>();
        for (int i = 0; i < wordList.size(); i++) {
            st.add(wordList.get(i));
        }
        st.remove(beginWord);
        while(!queue.isEmpty()) {
            Pair<String , Integer> entry = queue.poll();
            String currentWord = entry.first;
            int currentSteps = entry.second;
            if (currentWord.equals(endWord)) return currentSteps;

            for (int i = 0; i < currentWord.length(); i++) {
                for (char ch = 'a'; ch <= 'z'; ch++) {
                    char[] replacedCharArray = currentWord.toCharArray();
                    replacedCharArray[i] = ch;
                    String replacedWord = new String(replacedCharArray);

                    if (st.contains(replacedWord)) {
                        st.remove(replacedWord);
                        queue.add(new Pair<>(replacedWord, currentSteps + 1));
                    }
                }
            }
        }
        return 0;

    }

    public int calculate(String s) {
        int res = 0, sum = 0, sign = 1;
        Stack<Integer> myStack = new Stack<>();
        myStack.push(1);
        for (char ch : s.toCharArray()) {
            if (Character.isDigit(ch))
                sum = sum * 10 + (ch - '0');
            else {
                res += sum * sign * myStack.peek();
                sum = 0;
                if (ch == '-')
                    sign = -1;
                else if (ch == '+')
                    sign = 1;
                else if (ch == '(') {
                    myStack.push(myStack.peek() * sign);
                    sign = 1;
                } else if (ch == ')')
                    myStack.pop();
            }
        }
        return res += sign * sum;
    }

    public int calculate2(String s) {
        int sum = 0;
        int sign = 1;

        Stack<Integer> stack = new Stack<>();

        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (Character.isDigit(ch)) {
                int val = 0;
                while (i < s.length() && Character.isDigit(s.charAt(i))) {
                    val = val * 10 + (s.charAt(i) - '0');
                    i++;
                }
                i--;
                val = val * sign;
                sum += val;
                sign = 1;
            } else if (ch == '-') {
                sign = -1;
            } else if (ch == '(') {
                stack.push(sum);
                stack.push(sign);
                sum = 0;
                sign = 1;
            } else if (ch == ')') {
                sum = sum * stack.pop();
                sum = sum + stack.pop();
            }
        }

        return sum;
    }

    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int N = startTime.length;
        int[][] jobs = new int[N][3];
        for(int i=0; i < N; i++) {
            jobs[i] = new int[]{startTime[i], endTime[i], profit[i]};
        }

        Arrays.sort(jobs, (a, b) -> a[1] - b[1]);
        TreeMap<Integer, Integer> dp = new TreeMap<>();
        dp.put(0, 0);

        return 0;
    }

    public ListNode mergeKLists(ListNode[] lists) {
        if(lists.length == 0) {
            return null;
        }

        int N = lists.length - 1;
        while(N > 0) {
            int left = 0;
            int right = N;
            while(left < right) {
                lists[left] = merge2List(lists[left], lists[right]);
                left++;
                right--;
            }
            N = right;
        }
        
        return lists[0];
    }

    private ListNode merge2List(ListNode list1, ListNode list2) {
        if(list1 == null) {
            return list2;
        }
        if(list2 == null) {
            return list1;
        }

        ListNode head;
        if(list1.val < list2.val) {
            head = list1;
            list1 = list1.next;
        } else {
            head = list2;
            list2 = list2.next;
        }

        ListNode temp = head;
        while(list1 != null && list2 != null) {
            if(list1.val < list2.val) {
                temp.next = list1;
                list1 = list1.next;
            } else {
                temp.next = list2;
                list2 = list2.next;
            }
            temp = temp.next;
        }
        if(list1 != null) {
            temp.next = list1;
        }
        if(list2 != null) {
            temp.next = list2;
        }

        return head;
    }

    public ListNode reverseList(ListNode head) {
        if(head == null || head.next == null) {
            return head;
        }

        ListNode prev, curr, next;
        prev = null;
        curr = head;
        next = head;
        while(next != null) {
            next = next.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }

        return prev;

    }

    public int largestRectangleArea(int[] heights) {
        int maxArea = 0;
        Stack<Pair<Integer, Integer>> stack = new Stack<>(); // Pair<index, height>
        for(int i=0; i<heights.length; i++) {
            int start = i;
            while(!stack.isEmpty() && stack.peek().second > heights[i]) {
                Pair<Integer, Integer> pair = stack.pop();
                maxArea = Math.max(maxArea, pair.second * (i - pair.first));
                start = pair.first;
            }
            stack.push(new Pair<>(i, heights[i]));
        }
        while(!stack.isEmpty()) {
            Pair<Integer, Integer> pair = stack.pop();
            maxArea = Math.max(maxArea, pair.second * (heights.length - pair.first));
        }
        return maxArea;
    }

    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        char top;
        for(char c: s.toCharArray()) {
            if(c == '(' || c == '{' || c == '[') {
                stack.push(c);
            } else {
                if(stack.isEmpty()) {
                    return false;
                }
                top = stack.pop();
                if(c == ')') {
                    if(top != '(') {
                        return false;
                    }
                } else if(c == '}') {
                    if(top != '{') {
                        return false;
                    }
                } else if(c == ']') {
                    if(top != '[') {
                        return false;
                    }
                }
            }
        }
        return stack.isEmpty();
    }

    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if(list1 == null) {
            return list2;
        }
        if(list2 == null) {
            return list1;
        }
        ListNode head;
        if(list1.val < list2.val) {
            head = list1;
            list1 = list1.next;
        } else {
            head = list2;
            list2 = list2.next;
        }
        ListNode tmp = head;
        while(list1 != null && list2 != null) {
            if(list1.val < list2.val) {
                tmp.next = list1;
                list1 = list1.next;
            } else {
                tmp.next = list2;
                list2 = list2.next;
            }
            tmp = tmp.next;
        }
        if(list1 != null) {
            tmp.next = list1;
        }
        if(list2 != null) {
            tmp.next = list2;
        }

        return head;

    }

    int mod = (int)Math.pow(10, 9) + 7;
    public int numRollsToTarget(int n, int k, int target) {
        int[][] dp = new int[n+1][target+1];
        for(int[] d: dp) {
            Arrays.fill(d, -1);
        }
        return solveNumRollsToTarget(dp, n, k, target);

    }

    private int solveNumRollsToTarget(int[][] dp, int n, int k, int target) {
        if(target == 0 && n == 0) {
            return 1;
        }
        if(n == 0 || target <= 0) {
            return 0;
        }
        if(dp[n][target] != -1) {
            return (dp[n][target])%mod;
        }
        int result = 0;
        for(int i=1; i<=k; i++) {
            result = (result + solveNumRollsToTarget(dp, n-1, k, target - i))%mod;
        }

        return dp[n][target]=(result)%mod;
    }

    public String mergeAlternately(String word1, String word2) {
        StringBuilder result = new StringBuilder();
        int i = 0;
        while (i < word1.length() || i < word2.length()) {
            if (i < word1.length()) {
                result.append(word1.charAt(i));
            }
            if (i < word2.length()) {
                result.append(word2.charAt(i));
            }
            i++;
        }
        return result.toString();
    }

    public int maxProfit(int[] prices) {
        int result = 0;
        int left = 0;
        int right = 1;
        while(right < prices.length) {
            if(prices[right] > prices[left]) {
                result = Math.max(result, prices[right] - prices[left]);
            } else {
                left = right;
            }
            right++;
        }
        return result;

    }
    public boolean isPalindrome(String s) {
        if(s.isEmpty()) {
            return true;
        }
        int start = 0;
        int last = s.length() - 1;

        while(start <= last) {
            char firstChar = s.charAt(start);
            char lastChar = s.charAt(last);
            if(!Character.isLetterOrDigit(firstChar)) {
                start++;
            } else if (!Character.isLetterOrDigit(lastChar)) {
                last--;
            } else {
                if(Character.toLowerCase(firstChar) != Character.toLowerCase(lastChar)) {
                    return false;
                }
                start++;
                last--;
            }
        }
        return true;
    }
    public TreeNode invertTree(TreeNode root) {
        if(root == null) {
            return null;
        }

        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;

        invertTree(root.left);
        invertTree(root.right);

        return root;
    }
    public String gcdOfStrings(String str1, String str2) {
        if(! (str1 + str2).equals(str2 + str1)) {
            return "";
        } else {
            int gcd = gcd(str1.length(), str2.length());
            return str1.substring(0, gcd);
        }
    }

    private int gcd(int i, int j) {
        return j == 0 ? i : gcd(j, i%j);
    }

    public boolean isAnagram(String s, String t) {
        Map<Character, Integer> map = new HashMap<>();
        for(char c: s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        for(char c: t.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) - 1);
        }
        for(int count: map.values()) {
            if(count != 0) {
                return false;
            }
        }
        return true;
    }
    public int search(int[] nums, int target) {
        int start = 0;
        int end = nums.length - 1;
        while(start < end) {
            int mid = start + (end - start)/2;
            if(nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        return -1;
    }

    public int[][] floodFill(int[][] image, int sr, int sc, int color) {
        int initialColor = image[sr][sc];
        solveFloodFill(image, sr, sc, color, initialColor);
        return image;
    }

    private void solveFloodFill(int[][] image, int sr, int sc, int color, int initialColor) {
        if(sr > image.length - 1 || sr < 0 || sc > image[0].length - 1 || sc < 0) {
            return;
        }
        if(image[sr][sc] != initialColor || image[sr][sc] == color) {
            return;
        }
        image[sr][sc] = color;
        solveFloodFill(image, sr+1, sc, color, initialColor);
        solveFloodFill(image, sr, sc+1, color, initialColor);
        solveFloodFill(image, sr-1, sc, color, initialColor);
        solveFloodFill(image, sr, sc-1, color, initialColor);

    }

    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        List<Boolean> result = new ArrayList<>();
        int max = candies[0];
        for(int i = 0; i < candies.length; i++) {
            max = Math.max(max, candies[i]);
        }
        for(int i = 0; i < candies.length; i++) {
            if(candies[i] + extraCandies >= max) {
                result.add(true);
            } else {
                result.add(false);
            }
        }
        return result;
    }

    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        for (int i = 0; i < flowerbed.length && n > 0; i++) {
            if (flowerbed[i] == 0 && (i == 0 || flowerbed[i-1] == 0) && (i == flowerbed.length-1 || flowerbed[i+1] == 0)) {
                i++;
                n--;
            }
        }
        return n == 0;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        TreeNode curr = root;
        while(curr != null) {
            if(p.val < curr.val && q.val < curr.val) {
                curr = curr.left;
            } else if(p.val > curr.val && q.val > curr.val) {
                curr = curr.right;
            } else {
                return curr;
            }
        }
        return curr;
    }

    public boolean isBalanced(TreeNode root) {
        return height(root) != -1;
    }

    private int height(TreeNode root) {
        if(root == null) {
            return 0;
        }
        int left = height(root.left);
        if(left == -1) return -1;
        int right = height(root.right);
        if(right == -1) return -1;
        if(Math.abs(left - right) > 1) return -1;
        return 1 + Math.max(left, right);
    }
    public boolean hasCycle(ListNode head) {
        if(head == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head;
        while(fast.next != null) {
            slow = slow.next;
            if(slow == null) {
                return false;
            }
            fast = fast.next.next;
            if(fast.next == null) {
                return false;
            }
            if(slow == fast) {
                return true;
            }
        }
        return false;

    }

    public String reverseVowels(String s) {
        char[] word = s.toCharArray();
        int start = 0;
        int end = s.length() - 1;
        Set<Character> vowels = Set.of('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U');

        while (start < end) {
            while (start < end && !vowels.contains(word[start])) {
                start++;
            }
            while (start < end && !vowels.contains(word[end])) {
                end--;
            }

            // Swap the vowels
            char temp = word[start];
            word[start] = word[end];
            word[end] = temp;

            // Move the pointers towards each other
            start++;
            end--;
        }

        return new String(word);
    }

    public String reverseWords(String s) {
        s = s.replaceAll("\\s+"," ").trim();
        String[] strArr = s.split(" ");
        for(int i = 0; i < strArr.length/2; i++) {
            int j = strArr.length - i - 1;
            String temp = strArr[i];
            strArr[i] = strArr[j];
            strArr[j] = temp;
        }
        return String.join(" ", strArr);
    }

    public int climbStairs(int n) {
        if(n == 0 || n == 1 || n == 2) {
            return n;
        }
        int[] dp = new int[n+1];
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;

        for(int i=3; i<=n; i++) {
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }


    private boolean isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }

    public int longestPalindrome_2(String s) {
        int oddCount = 0;
        Map<Character, Integer> map = new HashMap<>();
        for(char ch: s.toCharArray()) {
            map.put(ch, map.getOrDefault(ch, 0) + 1);
            if(map.get(ch) %2 == 1) {
                oddCount++;
            } else {
                oddCount--;
            }
        }
        return oddCount > 1? s.length() - oddCount + 1: s.length();
    }

    public boolean increasingTriplet(int[] nums) {
        if(nums == null | nums.length < 3) {
            return false;
        }
        int a = Integer.MAX_VALUE;
        int b = Integer.MAX_VALUE;

        for(int num: nums) {
            if(a >= num) {
                a = num;
            } else if(b >= num) {
                b = num;
            } else {
                return true;
            }
        }
        return false;
    }

    public int compress(char[] chars) {
        int ans = 0;
        for (int i = 0; i < chars.length;) {
            final char letter = chars[i];
            int count = 0;

            while (i < chars.length && chars[i] == letter) {
                ++count;
                ++i;
            }
            chars[ans++] = letter;

             if (count > 1) {
                for (final char c : String.valueOf(count).toCharArray()) {
                    chars[ans++] = c;
                }
            }
        }
        return ans;
    }

    public String addBinary(String a, String b) {
        StringBuilder sb = new StringBuilder(); //Google immutability or string vs stringbuilder if you don't know why we use this instead of regular string
        int i = a.length() - 1, j = b.length() -1, carry = 0; //two pointers starting from the back, just think of adding two regular ints from you add from back
        while (i >= 0 || j >= 0) {
            int sum = carry; //if there is a carry from the last addition, add it to carry
            if (j >= 0) sum += b.charAt(j--) - '0'; //we subtract '0' to get the int value of the char from the ascii
            if (i >= 0) sum += a.charAt(i--) - '0';
            sb.append(sum % 2); //if sum==2 or sum==0 append 0 cause 1+1=0 in this case as this is base 2 (just like 1+9 is 0 if adding ints in columns)
            carry = sum / 2; //if sum==2 we have a carry, else no carry 1/2 rounds down to 0 in integer arithematic
        }
        if (carry != 0) sb.append(carry); //leftover carry, add it
        return sb.reverse().toString();
    }

    public int majorityElement(int[] nums) {
        int count = 0;
        int candidate = 0;

        for (int num : nums) {
            if (count == 0) {
                candidate = num;
            }

            if (num == candidate) {
                count++;
            } else {
                count--;
            }
        }

        return candidate;
    }

    // https://leetcode.com/problems/middle-of-the-linked-list
    public ListNode middleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    // https://leetcode.com/problems/maximum-depth-of-binary-tree/
    public int maxDepth(TreeNode root) {
        if(root == null) {
            return 0;
        }
        if(root.left == null && root.right == null) {
            return 1;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }

    // https://leetcode.com/problems/diameter-of-binary-tree/
    int diameterOfBinaryTree = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        dfs(root);
        return diameterOfBinaryTree;
    }

    private int dfs(TreeNode root) {
        if(root == null) {
            return -1;
        }
        int left = dfs(root.left);
        int right = dfs(root.right);

        diameterOfBinaryTree = Math.max(diameterOfBinaryTree, 2+left+right);

        return 1 + Math.max(left, right);
    }

    public void moveZeroes(int[] nums) {
        if(nums.length == 1) {
            return;
        }
        int left = 0;
        int right = 0;
        while(right < nums.length) {
            if(nums[right] != 0) {
                int temp = nums[left];
                nums[left] = nums[right];
                nums[right] = temp;
                left++;
            }
            right++;
        }
    }

    // https://leetcode.com/problems/is-subsequence/
    public boolean isSubsequence(String s, String t) {
        int i = 0;
        int j = 0;
        while(i < s.length() && j < t.length() - 1) {
            if(s.charAt(i) == t.charAt(j)) {
                j++;
            }
            i++;
        }
        return j == t.length() - 1;
    }

    // https://leetcode.com/problems/maximum-subarray/

    public int maxSubArray(int[] nums) {
        int maxSum = nums[0];
        int currSum = 0;
        for(int num: nums) {
            if(currSum < 0) {
                currSum = 0;
            }
            currSum = currSum + num;
            maxSum = Math.max(maxSum, currSum);
        }
        return maxSum;
    }

    // https://leetcode.com/problems/insert-interval/
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> result = new ArrayList<>();
        int j;

        for(int[] interval: intervals) {
            if(interval[1] < newInterval[0]) {
                result.add(interval);
            } else if (interval[0] > newInterval[1]) {
                result.add(newInterval);
                newInterval = interval;
            } else {
                newInterval[0] = Math.min(newInterval[0], interval[0]);
                newInterval[1] = Math.max(newInterval[1], interval[1]);
            }
        }
        result.add(newInterval);
        return result.toArray(new int[result.size()][]);

    }

    // https://leetcode.com/problems/max-number-of-k-sum-pairs
    public int maxOperations(int[] nums, int k) {
        HashMap<Integer,Integer>map=new HashMap<>();
        int count=0;
        for(int i=0;i<nums.length;i++){
            //to check if that k-nums[i] present and had some value left or already paired
            if(map.containsKey(k-nums[i]) && map.get(k-nums[i])>0){
                count++;
                map.put(k-nums[i],map.get(k-nums[i])-1);
            }else{
                //getOrDefault is a easy way it directly checks if value is 0 returns 0 where I added 1
                //and if some value is present then it returns that value "similar to map.get(i)" and I added 1 on it
                map.put(nums[i],map.getOrDefault(nums[i],0)+1);
            }
        }
        return count;
    }

    // https://leetcode.com/problems/01-matrix/description/
    public int[][] nearest(int[][] grid) {
        // Code here
        int rows = grid.length;
        int cols = grid[0].length;
        int[][] result = new int[rows][cols];

        for(int a[]: result) {
            Arrays.fill(a, -1);
        }

        Queue<int[]> queue = new LinkedList<>();
        int[][] directions = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(grid[i][j] == 1) {
                    result[i][j] = 0;
                    queue.add(new int[]{i, j});
                }
            }
        }

        while(!queue.isEmpty()) {
            int[]p = queue.poll();
            int i = p[0];
            int j = p[1];

            for(int[] d: directions) {
                int new_i = i + d[0];
                int new_j = j + d[1];

                if(new_i >= 0 && new_i < rows && new_j >= 0 && new_j < cols
                && result[new_i][new_j] == -1) {
                    result[new_i][new_j] = 1 + result[i][j];
                    queue.add(new int[]{new_i, new_j});
                }
            }
        }

        return result;

    }

    public void pq_test() {
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);

        pq.add(4);
        pq.add(3);
        pq.add(10);
        pq.add(1);

        while(!pq.isEmpty()) {
            System.out.println(pq.poll());
        }
    }

    // https://leetcode.com/problems/longest-substring-without-repeating-characters/
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        int maxLength = 0;
        int l = 0;

        for(int r = 0; r < s.length(); r++) {
            while(set.contains(s.charAt(r))) {
                set.remove(s.charAt(l));
                l++;
            }
            set.add(s.charAt(r));
            maxLength = Math.max(maxLength, r - l + 1);
        }
        return maxLength;
    }

    // https://leetcode.com/problems/maximum-average-subarray-i
    public double findMaxAverage(int[] nums, int k) {

        long currentSum = 0;
        for(int i=0; i < k; i++) {
            currentSum = currentSum + nums[i];
        }
        long maxSum = currentSum;
        for(int i=k; i < nums.length; i++) {
            currentSum = currentSum + nums[i] - nums[i-k];
            maxSum = Math.max(maxSum, currentSum);
        }

        return maxSum/1.0/k;
    }

    // https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length
    public int maxVowels(String s, int k) {

        char[] strArray = s.toCharArray();
        int curr = 0;
        for(int i=0; i< k; i++) {
            if(isVowel(strArray[i])) {
                curr++;
            }
        }
        int maxCount = curr;
        for(int i=k; i<s.length(); i++) {
            if(isVowel(strArray[i])) {
                curr++;
            }
            if(isVowel(strArray[i-k])) {
                curr--;
            }
            maxCount = Math.max(maxCount, curr);
        }

        return maxCount;

    }

   // https://leetcode.com/problems/3sum/description/
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);

        for(int i = 0; i < nums.length - 2; i++) {
            if(i > 0 && nums[i] == nums[i-1]) {
                continue;
            }
            int left = i+1;
            int right = nums.length - 1;
            while(left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if(sum == 0) {
                    result.add(Arrays.asList(nums[i], nums[left], nums[right]));

                    while(left < right && nums[left] == nums[left+1]) {
                        left++;
                    }
                    while(left < right && nums[right] == nums[right-1]) {
                        right--;
                    }
                    left++;
                    right--;
                } else if(sum < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }
        return result;

    }

    // https://leetcode.com/problems/binary-tree-level-order-traversal/
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> currentLevel = new ArrayList<>(levelSize);
            for(int i =0; i < levelSize; i++) {
                TreeNode curr = queue.poll();
                currentLevel.add(curr.val);
                if(curr.left != null) {
                    queue.add(curr.left);
                }
                if(curr.right != null) {
                    queue.add(curr.right);
                }
            }
            result.add(currentLevel);
        }
        return result;
    }

    // https://leetcode.com/problems/max-consecutive-ones-iii/
    public int longestOnes(int[] nums, int k) {
        int start = 0;
        int end = 0;
        int zeros = 0;

        while (end < nums.length) {
            if(nums[end] == 0) {
                zeros++;
            }
            end++;
            if(zeros > k) {
                if(nums[start] == 0) {
                    zeros--;
                }
                start++;
            }
        }

        return end - start;
    }

    // https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/
    public int longestSubarray(int[] nums) {
        int start = 0;
        int end = 0;
        int zeros = 0;

        while (end < nums.length) {
            if(nums[end] == 0) {
                zeros++;
            }
            end++;
            if(zeros > 1) {
                if(nums[start] == 0) {
                    zeros--;
                }
                start++;
            }
        }

        return end - start - 1;
    }

    // https://leetcode.com/problems/find-the-highest-altitude/
    public int largestAltitude(int[] gain) {
        int curr = 0;
        int max = 0;
        for(int num: gain) {
            curr = curr + num;
            max = Math.max(curr, max);
        }
        return max;
    }

    // https://leetcode.com/problems/find-pivot-index
    public int pivotIndex(int[] nums) {
        int sum = 0;
        int leftSum = 0;
        for(int i=0; i<nums.length; i++) {
            sum = sum+nums[i];
        }
        for(int i=0; i<nums.length; i++) {
            if(leftSum == sum - leftSum - nums[i]) {
                return i;
            }
            leftSum = leftSum + nums[i];
        }
        return -1;
    }

    // https://leetcode.com/problems/course-schedule
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<List<Integer>> adj = new ArrayList<>(numCourses);
        for(int i=0;i<numCourses;i++){
            adj.add(new ArrayList<>());
        }
        int[] preReqRequired = new int[numCourses];
        for(int i=0; i<prerequisites.length; i++) {
            adj.get(prerequisites[i][1]).add(prerequisites[i][0]);
            preReqRequired[prerequisites[i][0]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for(int i=0; i<numCourses; i++) {
            if(preReqRequired[i] == 0) {
                queue.add(i);
            }
        }
        List<Integer> topoOrder = new ArrayList<>();
        while(!queue.isEmpty()) {
            int curr = queue.poll();
            topoOrder.add(curr);
            for(int i=0; i < adj.get(curr).size(); i++) {
                preReqRequired[adj.get(curr).get(i)]--;
                if(preReqRequired[adj.get(curr).get(i)] == 0) {
                    queue.add(adj.get(curr).get(i));
                }
            }
        }

        return topoOrder.size() == numCourses;
    }

    // https://leetcode.com/problems/coin-change/description/
    public int[] productExceptSelf__(int[] nums) {
        // nums = [1,2,3,4]
        // Ans - [24,12,8,6]
        // 1st iteration - [1, 1, 2, 6]
        int curr = 1;
        int []output = new int[nums.length];
        for(int i = 0; i < nums.length; i++) {
            output[i] = curr;
            curr = curr * nums[i];
        }
        curr = 1;
        for(int i = nums.length - 1; i >= 0; i--) {
            output[i] = output[i] * curr;
            curr = curr * nums[i];
        }
        return output;
    }

    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount+1];
        Arrays.fill(dp, -1);
        int ans = solveCoinChange(coins, amount, dp);
        return  ans != Integer.MAX_VALUE ?
                ans : -1;
    }

    private int solveCoinChange(int[] coins, int amount, int[] dp) {
        if(amount == 0) {
            return 0;
        }
        if(amount < 0) {
            return Integer.MAX_VALUE;
        }
        if(dp[amount] != -1) {
            return dp[amount];
        }

        int minCoins = Integer.MAX_VALUE;

        for (int i=0; i<coins.length; i++) {
            int ans = solveCoinChange(coins, amount-coins[i], dp);
            if(ans != Integer.MAX_VALUE) {
                minCoins = Math.min(minCoins, 1+ans);
            }

        }

        return dp[amount] = minCoins;
    }

    // https://leetcode.com/problems/find-the-difference-of-two-arrays
    public List<List<Integer>> findDifference(int[] nums1, int[] nums2) {
        List<List<Integer>> ans = new ArrayList<>();
        ans.add(new ArrayList<>());
        ans.add(new ArrayList<>());

        Set<Integer> set1 = new HashSet<>();
        Set<Integer> set2 = new HashSet<>();

        for (int num: nums1) {
            set1.add(num);
        }
        for (int num: nums2) {
            set2.add(num);
        }

        for(int num : set1){
            if(!set2.contains(num)){ ans.get(0).add(num); }
        }
        for(int num : set2){
            if(!set1.contains(num)){ ans.get(1).add(num); }
        }

        return ans;
    }

    // https://leetcode.com/problems/unique-number-of-occurrences
    public boolean uniqueOccurrences(int[] arr) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num: arr) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        Set<Integer> set = new HashSet<>();
        for (int num: map.values()) {
            if (set.contains(num)) {
                return false;
            }
            set.add(num);
        }
        return true;
    }

    // https://leetcode.com/problems/determine-if-two-strings-are-close
    public boolean closeStrings(String word1, String word2) {
        if(word1.length() != word2.length()) {
            return false;
        }
        int[] w1 = new int[26];
        int[] w2 = new int[26];

        for(char ch: word1.toCharArray()) {
            w1[ch-'a']++;
        }
        for(char ch: word2.toCharArray()) {
            w2[ch-'a']++;
        }
        for(int i=0; i<26; i++) {
            if((w1[i] == 0 && w2[i] != 0) || (w1[i] != 0 && w2[i] == 0)) {
                return false;
            }
        }

        Arrays.sort(w1);
        Arrays.sort(w2);

        return Arrays.compare(w1, w2) == 0;
    }

    // https://leetcode.com/problems/equal-row-and-column-pairs
    public int equalPairs(int[][] grid) {
        int pair=0;
        Map<String, Integer> map = new HashMap<>();
        StringBuilder sb = new StringBuilder();
        for(int i=0; i<grid.length; i++) {
            sb.setLength(0);
            for(int j=0; j<grid[0].length; j++) {
                sb.append(grid[i][j]).append("#");
            }
            map.put(sb.toString(), map.getOrDefault(sb.toString(), 0) + 1);
        }
        for(int i=0; i<grid[0].length; i++) {
            sb.setLength(0);
            for(int j=0; j<grid.length; j++) {
                sb.append(grid[j][i]).append("#");
            }
            if(map.containsKey(sb.toString())) {
                pair = pair + map.get(sb.toString());
            }
        }
        return pair;
    }

    public int orangesRotting(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int time = 0;
        int freshOranges = 0;
        int[][] directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        Queue<int[]> queue = new LinkedList<>();
        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                if(grid[i][j] == 2) {
                    queue.add(new int[]{i, j});
                }
                if(grid[i][j] == 1) {
                    freshOranges++;
                }
            }
        }

        while(!queue.isEmpty()) {
            time++;
            for(int i=0; i<queue.size(); i++) {
                int[] p = queue.poll();
                for(int[] d: directions) {
                    int new_i = p[0] + d[0];
                    int new_j = p[1] + d[1];

                    if(new_i >= 0 && new_i < m
                    && new_j >= 0 && new_j < n
                    && grid[new_i][new_j] == 1) {
                        grid[new_i][new_j] = 2;
                        queue.add(new int[]{new_i, new_j});
                        freshOranges--;
                    }
                }
            }
        }
        return freshOranges == 0? time : -1;
    }

    // https://leetcode.com/problems/search-in-rotated-sorted-array/
    public int search_2(int[] nums, int target) {
        int low = 0;
        int high = nums.length;

        while (low < high) {
            int mid = (low + high) / 2;

            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] >= nums[0]) {
                 if (target < nums[mid] && target >= nums[0]) {
                    high = mid;
                } else {
                    low = mid + 1;
                }
            } else {
                if (target > nums[mid] && target <= nums[nums.length - 1]) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }

        }
        return -1;
    }

    // https://leetcode.com/problems/removing-stars-from-a-string
    public String removeStars(String s) {
        StringBuilder sb = new StringBuilder();
        Stack<Character> stack = new Stack<>();
        for(char ch: s.toCharArray()) {
            if(ch != '*') {
                stack.push(ch);
                sb.append(ch);
            } else {
                if(!stack.isEmpty()) {
                    stack.pop();
                    sb.setLength(sb.length() - 1);
                }
            }
        }
        return sb.toString();
    }

    // https://leetcode.com/problems/asteroid-collision
    public int[] asteroidCollision(int[] asteroids) {
        Stack<Integer> s = new Stack<>();
        for (int asteroid : asteroids) {
            if (asteroid > 0 || s.isEmpty()) {
                s.push(asteroid);
            } else {
                while (!s.isEmpty() && s.peek() > 0 && s.peek() < Math.abs(asteroid)) {
                    s.pop();
                }
                if (!s.isEmpty() && s.peek() == Math.abs(asteroid)) {
                    s.pop();
                } else {
                    if (s.isEmpty() || s.peek() < 0) {
                        s.push(asteroid);
                    }
                }
            }
        }
        int[] res = new int[s.size()];
        for (int i = s.size() - 1; i >= 0; i--) {
            res[i] = s.pop();
        }
        return res;
    }

    // https://leetcode.com/problems/decode-string
    public String decodeString(String s) {
        Stack<Character> stack = new Stack<>();

        for (char c : s.toCharArray()) {
            if (c != ']')
                stack.push(c);

            else {
                StringBuilder sb = new StringBuilder();
                while (!stack.isEmpty() && Character.isLetter(stack.peek()))
                    sb.insert(0, stack.pop());

                String sub = sb.toString(); // this is the string contained in [ ]
                stack.pop(); // Discard the '[';

                // step 2:
                // after that get the number of
                // times it should repeat from stack

                sb = new StringBuilder();
                while (!stack.isEmpty() && Character.isDigit(stack.peek()))
                    sb.insert(0, stack.pop());

                int count = Integer.valueOf(sb.toString()); // this is the number

                // step 3:
                // repeat the string within the [ ] count
                // number of times and push it back into stack

                while (count > 0) {
                    for (char ch : sub.toCharArray())
                        stack.push(ch);
                    count--;
                }
            }
        }

        // final fetching and returning the value in stack
        StringBuilder retv = new StringBuilder();
        while (!stack.isEmpty())
            retv.insert(0, stack.pop());

        return retv.toString();
    }

    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        Map<String, Set<String>> graph = new HashMap<>();
        Map<String, String> owner = new HashMap<>();

        for(List<String> account: accounts) {
            String userName = account.get(0);
            Set<String> neighbors = new HashSet<>(account);
            neighbors.remove(userName);

            for(int i = 1; i < account.size(); i++) {
                String email = account.get(i);
                if(!graph.containsKey(email)) {
                    graph.put(email, new HashSet<>());
                }
                graph.get(email).addAll(neighbors);
                owner.put(email, userName);
            }
        }

        Set<String> visited = new HashSet<>();
        List<List<String>> results = new ArrayList<>();
        // DFS search the graph;
        for (String email : owner.keySet()) {
            if (!visited.contains(email)) {
                List<String> result = new ArrayList<>();
                dfs(graph, email, visited, result);
                Collections.sort(result);
                result.add(0, owner.get(email));
                results.add(result);
            }
        }
        return results;
    }

    public void dfs(Map<String, Set<String>> graph, String email, Set<String> visited, List<String> list) {
        list.add(email);
        visited.add(email);
        for (String neighbor: graph.get(email)) {
            if (!visited.contains(neighbor)) {
                dfs(graph, neighbor, visited, list);
            }
        }
    }

    // https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/
    public ListNode deleteMiddle(ListNode head) {
        ListNode dummy = new ListNode(-1), prev = dummy, slow = head, fast = head;
        prev.next = head;
        while (fast != null && fast.next != null) {
            prev = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        prev.next = slow.next;
        return dummy.next;
    }

    // https://leetcode.com/problems/odd-even-linked-list
    public ListNode oddEvenList(ListNode head) {
        if(head == null) {
            return head;
        }
        ListNode odd = head;
        ListNode even = head.next;
        ListNode evenHead = even;

        while (even != null && even.next != null) {
            odd.next = odd.next.next;
            even.next = even.next.next;
            odd = odd.next;
            even = even.next;
        }
        odd.next = evenHead;
        return head;
    }

    public int pairSum(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        int maxVal = 0;

        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        ListNode nextNode, prev = null;
        while (slow != null) {
            nextNode = slow.next;
            slow.next = prev;
            prev = slow;
            slow = nextNode;
        }

        while (prev != null) {
            maxVal = Math.max(maxVal, head.val + prev.val);
            prev = prev.next;
            head = head.next;
        }

        return maxVal;
    }

    int good = 0;
    public int goodNodes(TreeNode root) {
        cal(root, Integer.MIN_VALUE);
        return good;
    }
    private void cal(TreeNode root, int max) {
        if(root == null) {
            return;
        }
        if(root.val >= max) {
            good++;
        }
        max = Math.max(max, root.val);
        cal(root.left, max);
        cal(root.right, max);
    }

    // https://leetcode.com/problems/path-sum-iii
    int total = 0;
    public int pathSum(TreeNode root, int targetSum) {
        if(root == null) {
            return 0;
        }
        helper(root, targetSum, 0);
        pathSum(root.left, targetSum);
        pathSum(root.right, targetSum);
        return total;
    }

    private void helper(TreeNode root, int sum, long curr) {
        if(root == null) {
            return;
        }
        curr += root.val;
        if(curr == sum) {
            total++;
        }
        helper(root.left, sum, curr);
        helper(root.right, sum, curr);
    }

    // link - https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree
    public int maxLevelSum(TreeNode root) {
        int[] ans = new int[2];
        ans[0] = Integer.MIN_VALUE;
        int maxLevel = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()) {
            int levelSize = queue.size();
            maxLevel++;
            int sum = 0;
            for(int i=0; i < levelSize; i++) {
                TreeNode curr = queue.poll();
                sum = sum + curr.val;
                if(curr.left != null) {
                    queue.add(curr.left);
                }
                if(curr.right != null) {
                    queue.add(curr.right);
                }
            }
            if(ans[0] < sum) {
                ans[0] = sum;
                ans[1] = maxLevel;
            }

        }
        return ans[1];
    }

    // link - https://leetcode.com/problems/search-in-a-binary-search-tree
    public TreeNode searchBST(TreeNode root, int val) {
        if(root == null || root.val == val) {
            return root;
        }
        if(root.val > val) {
            return searchBST(root.left, val);
        } else {
            return searchBST(root.right, val);
        }
    }

    // link - https://leetcode.com/problems/delete-node-in-a-bst
    public TreeNode deleteNode(TreeNode root, int key) {
        if(root == null) {
            return null;
        }
        if(key < root.val) {
            root.left = deleteNode(root.left, key);
            return root;
        } else if(key > root.val) {
            root.right = deleteNode(root.right, key);
            return root;
        } else {
            if(root.left==null){
                return root.right;
            }
            else if(root.right==null){
                return root.left;
            } else {
                TreeNode min = root.right;
                while (min.left != null) {
                    min = min.left;
                }
                root.val = min.val;
                root.right = deleteNode(root.right, min.val);
                return root;
            }
        }
    }

    // link - https://leetcode.com/problems/keys-and-rooms
    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        boolean[] vis = new boolean[rooms.size()];
        vis[0] = true;
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        int count = 1;
        while(!stack.isEmpty()) {
            for (int k: rooms.get(stack.pop())) {
                if(!vis[k]) {
                    stack.push(k);
                    vis[k] = true;
                    count++;
                }
            }
        }
        return rooms.size() == count;
    }

    // link - https://leetcode.com/problems/number-of-provinces/
    public int findCircleNum(int[][] M) {
        int N = M.length;
        boolean[]visited = new boolean[N];
        int count = 0;

        for(int i = 0; i < N ;i++){
            if(!visited[i]){
                count++;
                dfs(M,i,visited);
            }
        }
        return count;
    }


    private void dfs(int[][]M,int i,boolean[]visited){
        for(int j = 0 ; j < M[i].length ; j++){
            if(!visited[j] && M[i][j] != 0){
                visited[j] = true;
                dfs(M,j,visited);
            }
        }
    }

    // link - https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/
    int dfs(List<List<Integer>> al, boolean[] visited, int from) {
        int change = 0;
        visited[from] = true;
        for (int to : al.get(from))
            if (!visited[Math.abs(to)])
                change += dfs(al, visited, Math.abs(to)) + (to > 0 ? 1 : 0);
        return change;
    }
    public int minReorder(int n, int[][] connections) {
        List<List<Integer>> al = new ArrayList<>();
        for(int i = 0; i < n; ++i)
            al.add(new ArrayList<>());
        for (int[] c : connections) {
            al.get(c[0]).add(c[1]);
            al.get(c[1]).add(-c[0]);
        }
        return dfs(al, new boolean[n], 0);
    }

    // link - https://leetcode.com/problems/evaluate-division
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        Map<String, Map<String, Double>> graph = makeGraph(equations, values);

        double []ans = new double[queries.size()];

        // check for every Query
        // store it in ans array;
        for(int i = 0; i < queries.size(); i++){
            ans[i] = dfs(queries.get(i).get(0) , queries.get(i).get(1) , new HashSet<>(), graph);
        }
        return ans;
    }

    private  Map<String, Map<String, Double>> makeGraph(List<List<String>> e, double[] values){
        // build a graph
        // like a -> b = values[i]
        // and b -> a  = 1.0 / values[i];
        Map<String, Map<String, Double>> graph = new HashMap<>();
        String u, v;

        for(int i = 0; i < e.size(); i++){
            u = e.get(i).get(0);
            v = e.get(i).get(1);

            graph.putIfAbsent(u, new HashMap<>());
            graph.get(u).put(v, values[i]);

            graph.putIfAbsent(v, new HashMap<>());
            graph.get(v).put(u, 1/values[i]);

        }
        return graph;
    }

    public double dfs(String src, String dest, Set<String> visited, Map<String, Map<String, Double>> graph){
        // check the terminated Case
        // if string is not present in graph return -1.0;
        // like [a, e] or [x, x] :)
        if(!graph.containsKey(src))
            return -1.0;

        // simply say check src and dest are equal :) then return dest
        // store it in weight varaible;
        //case like [a,a] also handle
        if(graph.get(src).containsKey(dest)){
            return graph.get(src).get(dest);
        }

        visited.add(src);

        for(Map.Entry<String, Double> nbr : graph.get(src).entrySet()){
            if(!visited.contains(nbr.getKey())){
                double weight = dfs(nbr.getKey(), dest, visited, graph);

                // if weight is not -1.0(terminate case)
                // then mutliply it
                // like in querie   a -> c => 2 * 3 = 6
                if(weight != -1.0){
                    return nbr.getValue() * weight;
                }
            }
        }
        return -1.0;
    }

    // link - https://leetcode.com/problems/nearest-exit-from-entrance-in-maze
    public int nearestExit(char[][] maze, int[] entrance) {
        int rows = maze.length;
        int columns = maze[0].length;

        Queue<int[]> queue = new LinkedList<>();
        queue.offer(entrance);
        maze[entrance[0]][entrance[1]] = '+';

        int[][] directions = new int[][] {{0,1},{0,-1},{1,0},{-1,0}};

        int steps = 0;
        int x, y;
        while (!queue.isEmpty()) {
            steps++;

            int n = queue.size();
            for (int i = 0; i < n; i++) {
                int[] current = queue.poll();

                for (int[] direction : directions) {
                    x = current[0] + direction[0];
                    y = current[1] + direction[1];

                    if (x < 0 || x >= rows || y < 0 || y >= columns) continue;
                    if (maze[x][y] == '+') continue;

                    if (x == 0 || x == rows - 1 || y == 0 || y == columns - 1) return steps;

                    maze[x][y] = '+';
                    queue.offer(new int[]{x, y});
                }
            }
        }
        return -1;
    }
    // link - https://leetcode.com/problems/kth-largest-element-in-an-array/
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int num: nums) {
            pq.add(num);
            if (pq.size() > k) {
                pq.poll();
            }
        }

        return pq.poll();
    }

    // link - https://leetcode.com/problems/maximum-subsequence-score
    public long maxScore(int[] nums1, int[] nums2, int k) {
        long maxScore = 0;
        long sum1 = 0;

        int[][] pairs = new int[nums1.length][2];

        for(int i=0; i<nums1.length; i++) {
            pairs[i][0] = nums1[i];
            pairs[i][1] = nums2[i];
        }

        Arrays.sort(pairs, (a, b) -> b[1] - a[1]);
        PriorityQueue<Integer> pq = new PriorityQueue<>();

        for(int[] pair: pairs) {
            sum1 += pair[0];
            pq.offer(pair[0]);

            if(pq.size() > k) {
                sum1 -= pq.poll();
            }
            if(pq.size() == k) {
                maxScore = Math.max(maxScore, sum1 * pair[1]);
            }
        }

        return maxScore;
    }

    // link - https://leetcode.com/problems/total-cost-to-hire-k-workers
    public long totalCost(int[] costs, int k, int candidates) {
        int i = 0;
        int j = costs.length - 1;
        PriorityQueue<Integer> pq1 = new PriorityQueue<>();
        PriorityQueue<Integer> pq2 = new PriorityQueue<>();

        long ans = 0;
        while (k-- > 0) {
            while (pq1.size() < candidates && i <= j) {
                pq1.offer(costs[i++]);
            }
            while (pq2.size() < candidates && i <= j) {
                pq2.offer(costs[j--]);
            }

            int t1 = pq1.size() > 0 ? pq1.peek() : Integer.MAX_VALUE;
            int t2 = pq2.size() > 0 ? pq2.peek() : Integer.MAX_VALUE;

            if (t1 <= t2) {
                ans += t1;
                pq1.poll();
            } else {
                ans += t2;
                pq2.poll();
            }
        }
        return ans;
    }

    // link - https://leetcode.com/problems/successful-pairs-of-spells-and-potions
    public int[] successfulPairs(int[] spells, int[] potions, long success) {
        int n = spells.length;
        int m = potions.length;
        int[] pairs = new int[n];
        Arrays.sort(potions);
        for (int i = 0; i < n; i++) {
            int spell = spells[i];
            int left = 0;
            int right = m - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                long product = (long) spell * potions[mid];
                if (product >= success) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            pairs[i] = m - left;
        }
        return pairs;
    }

    // link - https://leetcode.com/problems/house-robber/
    public int rob(int[] nums) {
        int dp[] = new int[nums.length+1];
        Arrays.fill(dp, -1);
        return solveRob(nums, 0, dp);
    }

    private int solveRob(int[] nums, int index, int[] dp) {
        if(index >= nums.length) {
            return 0;
        }
        if(dp[index] != -1) {
            return dp[index];
        }
        return dp[index] = Math.max(nums[index] + solveRob(nums, index+2, dp),
                solveRob(nums, index+1, dp));
    }

    // link - https://leetcode.com/problems/domino-and-tromino-tiling
    public int numTilings(int n) {
        Map<String, Integer> memo = new HashMap<>();
        return numTilingsForFinalStateOf(n, 2, memo);
    }
    public int numTilingsForFinalStateOf(int n, int colState, Map<String, Integer> memo) {
        int mod = 1_000_000_007;
        if (n < 0) return 0;
        if (n == 0) {
            if (colState == 2) {
                return 1;
            } else {
                return 0;
            }
        }
        String memKey = n+"|"+colState;
        if (memo.containsKey(memKey)) return memo.get(memKey);

        long ways = 0;
        if (colState == 0) {
            ways += numTilingsForFinalStateOf(n-2, 2, memo);
            ways += numTilingsForFinalStateOf(n-1, 1, memo);
        } else if (colState == 1) {
            ways += numTilingsForFinalStateOf(n-2, 2, memo);
            ways += numTilingsForFinalStateOf(n-1, 0, memo);
        } else if (colState == 2) {
            ways += numTilingsForFinalStateOf(n-1, 0, memo);
            ways += numTilingsForFinalStateOf(n-1, 1, memo);
            ways += numTilingsForFinalStateOf(n-1, 2, memo);
            ways += numTilingsForFinalStateOf(n-2, 2, memo);
        }

        ways = Math.floorMod(ways, mod);
        memo.put(memKey, (int) ways);
        return (int) ways;
    }

    // link - https://leetcode.com/problems/longest-common-subsequence
    public int longestCommonSubsequence(String text1, String text2) {
        int [][]dp = new int[text1.length()+1][text1.length()+1];
        for(int a[]: dp) {
            Arrays.fill(a, -1);
        }
        return solveLongestCommonSubsequence(text1, text2, 0, 0, dp);
    }

    private int solveLongestCommonSubsequence(String text1, String text2, int i, int j, int[][] dp) {
        if(i == text1.length() || j == text2.length()) {
            return 0;
        }
        if(dp[i][j] != -1) {
            return dp[i][j];
        }
        if(text1.charAt(i) == text2.charAt(j)) {
            return dp[i][j] = 1 + solveLongestCommonSubsequence(text1, text2, i+1, j+1, dp);
        } else {
            return dp[i][j] = Math.max(solveLongestCommonSubsequence(text1, text2, i+1, j, dp),
                    solveLongestCommonSubsequence(text1, text2, i, j+1, dp));
        }
    }

    // link - https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/
    public int maxProfit(int[] prices, int fee) {
        int[][] dp = new int[prices.length + 1][2];
        for(int a[]: dp) {
            Arrays.fill(a, -1);
        }
        return solveMaxProfit(prices, fee, 0, true, dp);
    }

    private int solveMaxProfit(int[] prices, int fee, int index, boolean buy, int[][] dp) {
        if(index == prices.length) {
            return 0;
        }
        if(dp[index][buy?1:0] != -1) {
            return dp[index][buy?1:0];
        }
        int profit = 0;
        if(buy) {
            profit = Math.max(-prices[index] + solveMaxProfit(prices, fee, index+1, false, dp) - fee,
                    solveMaxProfit(prices, fee, index + 1, true, dp));
        } else {
            profit = Math.max(prices[index] + solveMaxProfit(prices, fee, index+1, true, dp) - fee,
                    solveMaxProfit(prices, fee, index + 1, false, dp));

        }
        dp[index][buy?1:0] = profit;
        return dp[index][buy?1:0];
    }

    // problem link - https://leetcode.com/problems/edit-distance
    public int minDistance(String word1, String word2) {
        int[][] dp = new int[word1.length()][word2.length()];
        for(int a[]: dp) {
            Arrays.fill(a, -1);
        }
        return solveMinDistance(word1, word2, 0, 0, dp);
    }

    private int solveMinDistance(String word1, String word2, int i, int j, int[][] dp) {
        if(word1.length() == i) {
            return word2.length() - j;
        }
        if(word2.length() == j) {
            return word1.length() - i;
        }
        if(dp[i][j] != -1) {
            return dp[i][j];
        }
        if(word1.charAt(i) == word2.charAt(j)) {
            return dp[i][j] = solveMinDistance(word1, word2, i + 1, j + 1, dp);
        } else {
            //insert
            int insertAns = 1 + solveMinDistance(word1, word2, i, j+1, dp);
            //delete
            int deleteAns = 1 + solveMinDistance(word1, word2, i+1, j, dp);
            //replace
            int replaceAns = 1 + solveMinDistance(word1, word2, i+1, j+1, dp);

            return dp[i][j] = Math.min(insertAns, Math.min(deleteAns, replaceAns));
        }
    }

    // problem link - https://leetcode.com/problems/search-suggestions-system
    public List<List<String>> suggestedProducts(String[] products, String searchWord) {
        PriorityQueue<String> pq = new PriorityQueue<>(3, (s1,s2) -> s1.compareTo(s2));
        List<List<String>> list = new ArrayList<>();

        for(int i = 1; i<=searchWord.length(); i++){
            String temp = searchWord.substring(0, i);
            for(String s : products){
                if(s.startsWith(temp)){
                    pq.offer(s);
                }
            }
            List<String> temp_list = new ArrayList<>();
            for(int j = 0; j<3; j++){
                if(pq.peek() != null){
                    temp_list.add(pq.poll());
                }
            }
            pq.clear();
            list.add(temp_list);
        }
        return list;
    }

    // problem link - https://leetcode.com/problems/jump-game
    public boolean canJump(int[] nums) {
        int reachable = 0;
        for(int i=0; i<nums.length; i++) {
            if(i > reachable) {
                return true;
            }
            reachable = Math.max(reachable, i + nums[i]);
        }
        return true;
    }

    // problem link - https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string
    public int strStr(String haystack, String needle) {
        int j = 0;
        int index = 0;
        for(int i=0; i<haystack.length(); i++) {
            if(haystack.charAt(i) != needle.charAt(j)) {
                j= 0;
                i = index;
                index = i + 1;
            } else {
                j++;
            }
            if(j == needle.length()) {
                return index;
            }
        }
        return -1;
    }


    public static void main(String arrr[]) {
        Solution solution = new Solution();
    }
}

