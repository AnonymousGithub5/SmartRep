function FunctionDefinition_0 ( address _from , bytes32 Parameter_0 ) public { UserDefinedTypeName_0 memory VariableDeclaration_0 = Identifier_0 ( _from ) ; require ( Identifier_1 . amount <= token . MemberAccess_0 ( msg . sender , address ) ) ; require ( Identifier_2 [ _from ] [ msg . sender ] . MemberAccess_1 == 0 ) ; require ( Identifier_3 [ _from ] [ msg . sender ] + 3 days <= now ) ; token . transferFrom ( msg . sender , address ( this ) , Identifier_4 . amount ) ; Identifier_5 [ _from ] [ msg . sender ] = Identifier_6 ( block . number , now , Identifier_7 , Identifier_8 ) ; emit Identifier_9 ( _from , msg . sender ) ; }