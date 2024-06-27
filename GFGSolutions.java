
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
	public String FirstNonRepeating(String A)
	    {
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
}
