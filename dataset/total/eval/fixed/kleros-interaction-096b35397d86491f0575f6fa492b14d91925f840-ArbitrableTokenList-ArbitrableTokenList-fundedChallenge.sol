function FunctionDefinition_0 ( bytes32 Parameter_0 ) external payable { UserDefinedTypeName_0 storage VariableDeclaration_0 = Identifier_0 [ _tokenID ] ; require ( item . MemberAccess_0 != address ( 0 ) , stringLiteral_0 ) ; require ( ! item . disputed || arbitrator . MemberAccess_1 ( item . disputeID ) == Identifier_1 . MemberAccess_2 . MemberAccess_3 , stringLiteral_1 ) ; require ( item . status == Identifier_2 . MemberAccess_4 || item . status == Identifier_3 . MemberAccess_5 , stringLiteral_2 ) ; require ( msg . value >= item . MemberAccess_6 + arbitrator . MemberAccess_7 ( Identifier_4 ) , stringLiteral_3 ) ; item . balance += item . MemberAccess_8 ; item . MemberAccess_9 = msg . value - item . MemberAccess_10 ; item . MemberAccess_11 = msg . sender ; item . MemberAccess_12 = now ; emit Identifier_5 ( _tokenID , msg . sender , item . MemberAccess_13 ) ; }