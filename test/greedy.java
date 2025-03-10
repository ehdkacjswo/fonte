/**
 * This is a Javadoc comment.
 * It describes the class.
 */
public class Sample {

    @Deprecated // Annotation before method
    public void method1() {
        System.out.println("This is a string, not a comment: /* real comment? */");
        char c = '\u03a9'; // A character literal

        // Single-line comment with annotation: @SuppressWarnings("unused")
        int number = 42; /* Block comment before declaration */
    }

    /**
     * Another Javadoc comment.
     * Contains an annotation inside.
     * @Generated("tool")
     */
    @Override
    public String method2() {
        /* A block comment with a Javadoc inside
         * /**
         *  * Not a real Javadoc
         *  */
        */
        return "@This_is_not_an_annotation";
    }

    @SuppressWarnings("unchecked") // Annotation with parameter
    private List<String> method3() {
        String fake$Comment = "// This is not a comment";
        String multilineFakeComment = "/* Not a real block comment */";
        return new ArrayList<>();
    }

    /* Nested block comment test
     /* This should be ignored */
     int testValue = 100;
     */

    @CustomAnnotation // Custom annotation
    class InnerClass {
        int value = 10;
    }
}
