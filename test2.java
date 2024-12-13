import java.util.ArrayList;
import java.util.List;

public class TaskManager {
    private List<String> tasks;

    public TaskManager() {
        tasks = new ArrayList<>();
    }

    // Adds a new task, returns success status
    public boolean addTask(String task) {
        if (task == null || task.trim().isEmpty()) {
            System.out.println("Cannot add an empty task.");
            return false;
        }
        tasks.add(task);
        return true;
    }

    // Removes a task by name, returns success status
    public boolean removeTask(String task) {
        if (tasks.remove(task)) {
            System.out.println("Removed task: " + task);
            return true;
        } else {
            System.out.println("Task not found: " + task);
            return false;
        }
    }

    // Lists all tasks with indices
    public void listTasks() {
        if (tasks.isEmpty()) {
            System.out.println("No tasks available.");
        } else {
            for (int i = 0; i < tasks.size(); i++) {
                System.out.println((i + 1) + ". " + tasks.get(i));
            }
        }
    }

    // Updates a task by index
    public boolean updateTask(int index, String newTask) {
        if (index < 1 || index > tasks.size()) {
            System.out.println("Invalid task index.");
            return false;
        }
        if (newTask == null || newTask.trim().isEmpty()) {
            System.out.println("New task cannot be empty.");
            return false;
        }
        tasks.set(index - 1, newTask);
        return true;
    }

    public static void main(String[] args) {
        TaskManager manager = new TaskManager();
        manager.addTask("Finish homework");
        manager.addTask("Buy groceries");
        manager.addTask(""); // Invalid task
        manager.listTasks();
        manager.updateTask(2, "Buy vegetables");
        manager.removeTask("Finish homework");
        manager.listTasks();
    }
}
