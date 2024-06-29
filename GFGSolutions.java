
class GFGSolutions
{
    static class Position {
        int x;
        int y;
        Position(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }
    static class BFSElement {
    	int i, j;
    	BFSElement(int i, int j) {
		this.i = i;
		this.j = j;
    	}
    }
    // problem link - https://www.geeksforgeeks.org/problems/rotten-oranges2536/1
    public int orangesRotting(int[][] grid)
    {
        // Code here
        if(grid == null || grid.length == 0) {
            return 0; 
        }
        int total = 0, rotten = 0, time = 0;
        Queue<Position> queue = new LinkedList<>();

        int[][] dir = new int[][]{};
        for (int i=0; i<grid.length; i++) {
            for (int j=0; j<grid[0].length; j++) {
                if (grid[i][j] == 2) {
                    queue.add(new Position(i, j));
                }
                if (grid[i][j] != 0) {
                    total++;
                }
            }
        }
        
        if (total == 0) {
            return 0;
        }
        
        while (!queue.isEmpty() && rotten < total) {
            int size = queue.size();
            rotten = rotten + size;
            if (rotten == total) {
                return time;
            }
            time++;
            for (int i=0; i < size; i++) {
                Position p = queue.poll();
                
                if (p.x + 1 < grid.length && grid[p.x+1][p.y] == 1) {
                    grid[p.x+1][p.y] = 2;
                    queue.add(new Position(p.x+1, p.y));
                }
                
                if (p.y + 1 < grid[0].length && grid[p.x][p.y+1] == 1) {
                    grid[p.x][p.y+1] = 2;
                    queue.add(new Position(p.x, p.y+1));
                }
                
                if (p.x - 1 >= 0 && grid[p.x-1][p.y] == 1) {
                    grid[p.x-1][p.y] = 2;
                    queue.add(new Position(p.x-1, p.y));
                }
                
                if (p.y - 1 >=0 && grid[p.x][p.y-1] == 1) {
                    grid[p.x][p.y-1] = 2;
                    queue.add(new Position(p.x, p.y-1));
                }
            }
        }
        return -1;
    }

    //problem link - https://www.geeksforgeeks.org/problems/longest-consecutive-subsequence2449/1
	static int findLongestConseqSubseq(int arr[], int N)
	{
	   // add your code here
	   if (N == 0) {
	       return 0;
	   }
	   Set<Integer> set = new HashSet<>();
	   int longest = 1;
	   for (int num: arr) {
	       set.add(num);
	   }
	   for (int num: set) {
	       if (!set.contains(num-1)) {
	           int count = 1;
	           int curr = num;
	           while (set.contains(curr+1)) {
	               curr++;
	               count++;
	           }
	           longest = Math.max(longest, count);
	       }
	   }
	   return longest;
	}
	
	// problem link - https://www.geeksforgeeks.org/problems/first-non-repeating-character-in-a-stream1216/1
	public String FirstNonRepeating(String A){
	        // code here
	        ArrayList<Character> list = new ArrayList<>(); 
	        HashMap<Character, Integer> map = new HashMap<>();
	        StringBuilder sb = new StringBuilder();
	 
	        for (char ch : A.toCharArray()) {
	            if (!map.containsKey(ch)) { 
	                list.add(ch);
	                map.put(ch, 1);
	            }
	            else {
	                int index = list.indexOf(ch);
	  
	                if (index != -1) 
	                      list.remove(index);
	            }
	            sb.append(list.isEmpty() ? '#' : list.get(0));
	        }
	        return sb.toString();
	    }

	// problem link - https://www.geeksforgeeks.org/problems/array-pair-sum-divisibility-problem3257/1
	 public boolean canPair(int[] nums, int k) {
		 if (nums.length % 2 != 0) {
			 return false; 
		 }
		 int freq[] = new int[k];
		 for (int num: nums) {
			 int y = num % k;
			 if (freq[(k-y)%k] != 0) {
				 freq[(k-y)%k]--
			 } else {
				freq[k]++; 
			 }
		 }
		 for (int i: freq) {
			 if (i != 0) {
				 return false;
			 }
		 }
		 return true;
	 }

	// problem link - https://www.geeksforgeeks.org/problems/path-in-matrix3805/1
    static int maximumPath(int N, int Matrix[][]) {

        int m = Matrix[0].length;

        int dp[][] = new int[N][m];
        for (int row[] : dp)
            Arrays.fill(row, -1);

        int maxi = Integer.MIN_VALUE;

        for (int j = 0; j < m; j++) {
            int ans = getMaxUtil(N - 1, j, m, Matrix, dp);
            maxi = Math.max(maxi, ans);
        }

        return maxi;
    }
    static int getMaxUtil(int i, int j, int m, int[][] matrix, int[][] dp) {
        // Base Conditions
        if (j < 0 || j >= m)
            return (int) Math.pow(-10, 9);
        if (i == 0)
            return matrix[0][j];

        if (dp[i][j] != -1)
            return dp[i][j];

        // Calculate three possible paths: moving up, left diagonal, and right diagonal
        int up = matrix[i][j] + getMaxUtil(i - 1, j, m, matrix, dp);
        int leftDiagonal = matrix[i][j] + getMaxUtil(i - 1, j - 1, m, matrix, dp);
        int rightDiagonal = matrix[i][j] + getMaxUtil(i - 1, j + 1, m, matrix, dp);

        // Store the maximum of the three paths in dp
        return dp[i][j] = Math.max(up, Math.max(leftDiagonal, rightDiagonal));
    }

// problem link - https://www.geeksforgeeks.org/problems/find-whether-path-exist5238/1
	public boolean is_Possible(int[][] grid) {
	
	        Queue<BFSElement> q = new LinkedList<>();
	        int R = grid.length;
	        int C = grid[0].length;
	        
	        for (int i = 0; i < R; ++i) {
	            for (int j = 0; j < C; ++j) {
	                if (grid[i][j] == 1) {
	                    q.add(new BFSElement(i, j));
	                    break;
	                }
	            }
	        }
	        while (q.size() != 0) {
	            BFSElement x = q.peek();
	            q.remove();
	            int i = x.i;
	            int j = x.j;
	
	            if (i < 0 || i >= R || j < 0 || j >= C) {
	                continue;
	            }
	 
	            if (grid[i][j] == 0) {
	                continue;
	            }
	            
	            if (grid[i][j] == 2) {
	                return true;
	            }
	 
	            // marking as wall upon successful visitation
	            grid[i][j] = 0;
	 
	            // pushing to queue u=(i,j+1),u=(i,j-1)
	            // u=(i+1,j),u=(i-1,j)
	            for (int k = -1; k <= 1; k += 2) {
	                q.add(new BFSElement(i + k, j));
	                q.add(new BFSElement(i, j + k));
	            }
	        }
	        return false;
    }

// problem link - https://www.geeksforgeeks.org/problems/implement-atoi/1
    public int atoi(String s) {
        int res = 0;
        int sign = 1;
        int i = 0;
        if (s.charAt(0) == '-') {
            sign = -1;
            i++;
        }
        for (; i < s.length(); ++i) {
            if (!Character.isDigit(s.charAt(i))) {
                return -1;
            }
             res = res * 10 + s.charAt(i) - '0';
        }
           
        return sign * res;
    }
}
