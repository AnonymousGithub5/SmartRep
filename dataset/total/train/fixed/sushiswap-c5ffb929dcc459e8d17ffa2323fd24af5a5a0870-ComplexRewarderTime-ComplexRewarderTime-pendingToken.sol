function FunctionDefinition_0 ( uint256 Parameter_0 , address _user ) public view returns ( uint256 Parameter_1 ) { PoolInfo memory pool = poolInfo [ _pid ] ; UserInfo storage user = userInfo [ _pid ] [ _user ] ; uint256 VariableDeclaration_0 = pool . MemberAccess_0 ; uint256 VariableDeclaration_1 = Identifier_0 ( Identifier_1 ) . MemberAccess_1 ( _pid ) . balanceOf ( Identifier_2 ) ; if ( block . timestamp > pool . MemberAccess_2 && Identifier_3 != 0 ) { uint256 VariableDeclaration_2 = block . timestamp . sub ( pool . MemberAccess_3 ) ; uint256 VariableDeclaration_3 = Identifier_4 . mul ( Identifier_5 ) . mul ( pool . MemberAccess_4 ) / Identifier_6 ; Identifier_7 = Identifier_8 . add ( Identifier_9 . mul ( Identifier_10 ) / Identifier_11 ) ; } pending = ( user . amount . mul ( Identifier_12 ) / Identifier_13 ) . sub ( user . rewardDebt ) . add ( user . MemberAccess_5 ) ; }