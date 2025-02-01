import java.lang.annotation.ElementType;
import java.lang.annotation.Target;
import java.util.List;
import java.util.function.Function;

// Annotation Type Declaration
@Target(ElementType.TYPE)
@interface MyAnnotation {
    String value();
}

// Interface Declaration with multiple type parameters
interface BaseInterface1<T, U> {
    void baseMethod1(T t, U u);
}

interface BaseInterface2<V> {
    V baseMethod2();
}

// Interface that extends multiple interfaces with type parameters
interface CombinedInterface<X, Y, Z> extends BaseInterface1<X, Y>, BaseInterface2<Z> {
    void combinedMethod(X x, Y y, Z z);
}

// Class Declaration that extends a superclass and implements multiple interfaces with type parameters
@MyAnnotation("TestClass")
class MyClass<A, B> extends AbstractBase implements CombinedInterface<A, B, String> {
    @Override
    public void baseMethod1(A a, B b) {
        System.out.println("BaseMethod1: " + a + ", " + b);
    }

    @Override
    public String baseMethod2() {
        return "BaseMethod2 result";
    }

    @Override
    public void combinedMethod(A a, B b, String z) {
        System.out.println("CombinedMethod: " + a + ", " + b + ", " + z);
    }

    // Method Declaration with multiple type arguments
    public <T, U> List<U> transform(List<T> input, Function<T, U> transformer) {
        return input.stream().map(transformer).toList();
    }

    public void demonstrate() {
        // Class instance creation with multiple type parameters
        MyClass<Integer, String> instance = new MyClass<>();

        // Normal method invocation with multiple type arguments
        List<String> result = instance.<Integer, String>transform(List.of(1, 2, 3), Object::toString);
        System.out.println("Transformed List: " + result);

        // Method reference with multiple type parameters
        Function<Integer, String> methodRef = this::<Integer, String>convert;
        System.out.println("Converted: " + methodRef.apply(42));

        // Super method invocation with qualifier and multiple type arguments
        String superResult = super.<Integer, String>superMethod(123);
        System.out.println("Super Method Result: " + superResult);
    }

    public <T, U> U convert(T input) {
        return (U) input.toString();
    }
}

// Abstract base class with a generic method
abstract class AbstractBase {
    public <T, U> U superMethod(T input) {
        return (U) ("SuperMethod: " + input.toString());
    }
}

public class TestASTParser {
    public static void main(String[] args) {
        MyClass<Integer, String> myClass = new MyClass<>();
        myClass.demonstrate();
    }
}
