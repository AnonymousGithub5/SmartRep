function FunctionDefinition_0 ( uint Parameter_0 ) public { UserDefinedTypeName_0 storage VariableDeclaration_0 = Identifier_0 [ Identifier_1 ] ; require ( task . status < Status . MemberAccess_0 , stringLiteral_0 ) ; require ( now - task . MemberAccess_1 > task . MemberAccess_2 , stringLiteral_1 ) ; task . status = Status . MemberAccess_3 ; uint amount = task . MemberAccess_4 [ uint ( Party . Requester ) ] + task . MemberAccess_5 [ uint ( Party . MemberAccess_6 ) ] ; task . parties [ uint ( Party . Requester ) ] . send ( amount ) ; task . MemberAccess_7 [ uint ( Party . Requester ) ] = 0 ; task . MemberAccess_8 [ uint ( Party . MemberAccess_9 ) ] = 0 ; }