function transferFrom ( ) external payable { if ( msg . data . length > 0 ) { address VariableDeclaration_0 = Identifier_0 [ msg . MemberAccess_0 ] ; if ( Identifier_1 == address ( 0 ) ) { emit Identifier_2 ( msg . value , msg . sender , msg . data ) ; } else { require ( Identifier_3 [ Identifier_4 ] , stringLiteral_0 ) ; assembly { AssemblyExpression_1 ( 0 , 0 , AssemblyExpression_0 ( ) ) let result := AssemblyExpression_5 ( AssemblyExpression_2 , AssemblyExpression_3 , 0 , AssemblyExpression_4 ( ) , 0 , 0 ) AssemblyExpression_7 ( 0 , 0 , AssemblyExpression_6 ( ) ) switch AssemblyExpression_8 case 0 { AssemblyExpression_10 ( 0 , AssemblyExpression_9 ( ) ) } default { AssemblyExpression_12 ( 0 , AssemblyExpression_11 ( ) ) } } } } }