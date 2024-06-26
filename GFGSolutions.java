
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
}
