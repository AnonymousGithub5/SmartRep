function FunctionDefinition_0 ( uint Parameter_0 , uint Parameter_1 , bytes32 Parameter_2 ) external ModifierInvocation_0 ( _disputeID , Identifier_3 . MemberAccess_4 ) { UserDefinedTypeName_0 storage VariableDeclaration_0 = disputes [ _disputeID ] ; require ( dispute . votes [ dispute . votes . length - 1 ] [ Identifier_0 ] . MemberAccess_0 == msg . sender , stringLiteral_0 ) ; dispute . votes [ dispute . votes . length - 1 ] [ Identifier_1 ] . MemberAccess_1 = Identifier_2 ; dispute . MemberAccess_2 [ dispute . MemberAccess_3 . length - 1 ] ++ ; }