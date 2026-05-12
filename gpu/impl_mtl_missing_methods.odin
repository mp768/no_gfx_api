#+build darwin 
#+private
package gpu

import "base:intrinsics"
import mtl "vendor:darwin/Metal"
import ca "vendor:darwin/QuartzCore"
import ns "core:sys/darwin/Foundation"
import cf "core:sys/darwin/CoreFoundation"

objcMsgSend :: intrinsics.objc_send

// Residency Set Descriptor

@(objc_class="MTLResidencySetDescriptor")
MTLResidencySetDescriptor :: struct { using _: ns.Copying(MTLResidencySetDescriptor) }

@(objc_type=MTLResidencySetDescriptor, objc_name="alloc", objc_is_class_method=true)
MTLResidencySetDescriptor_alloc :: #force_inline proc "c" () -> ^MTLResidencySetDescriptor {
	return objcMsgSend(^MTLResidencySetDescriptor, MTLResidencySetDescriptor, "alloc")
}

@(objc_type=MTLResidencySetDescriptor, objc_name="init")
MTLResidencySetDescriptor_init :: #force_inline proc "c" (self: ^MTLResidencySetDescriptor) -> ^MTLResidencySetDescriptor {
	return objcMsgSend(^MTLResidencySetDescriptor, self, "init")
}

@(objc_type=MTLResidencySetDescriptor, objc_name="label")
MTLResidencySetDescriptor_label :: #force_inline proc "c" (self: ^MTLResidencySetDescriptor) -> ^ns.String {
    return objcMsgSend(^ns.String, self, "label")
}

@(objc_type=MTLResidencySetDescriptor, objc_name="setLabel")
MTLResidencySetDescriptor_setLabel :: #force_inline proc "c" (self: ^MTLResidencySetDescriptor, label: ^ns.String) {
    objcMsgSend(nil, self, "setLabel:", label)
}

@(objc_type=MTLResidencySetDescriptor, objc_name="initialCapacity")
MTLResidencySetDescriptor_initialCapacity :: #force_inline proc "c" (self: ^MTLResidencySetDescriptor) -> ns.Integer {
    return objcMsgSend(ns.Integer, self, "initialCapacity")
}

@(objc_type=MTLResidencySetDescriptor, objc_name="setInitialCapacity")
MTLResidencySetDescriptor_setInitialCapacity :: #force_inline proc "c" (self: ^MTLResidencySetDescriptor, cap: ns.Integer) {
    objcMsgSend(nil, self, "setInitialCapacity:", cap)
}

// Residency Set

@(objc_class="MTLResidencySet")
MTLResidencySet :: struct { using _: ns.Object }

@(objc_type=MTLResidencySet, objc_name="addAllocation")
// NOTE(MP): The parameter being `^mtl.Resource` is technically wrong (even though that does inherit from MTLAllocation)
// but it works for our purposes since as long as we use it correctly, it shouldn't be a problem.
// 
// Additionally, I simply don't want to go through additional loops to get this working.
MTLResidencySet_addAllocation :: #force_inline proc "c" (self: ^MTLResidencySet, allocation: ^mtl.Resource) {
    objcMsgSend(nil, self, "addAllocation:", allocation)
}

@(objc_type=MTLResidencySet, objc_name="removeAllAllocations")
MTLResidencySet_removeAllAllocations :: #force_inline proc "c" (self: ^MTLResidencySet) {
    objcMsgSend(nil, self, "removeAllAllocations")
}

@(objc_type=MTLResidencySet, objc_name="removeAllocation")
MTLResidencySet_removeAllocation :: #force_inline proc "c" (self: ^MTLResidencySet, allocation: ^mtl.Resource) {
    objcMsgSend(nil, self, "removeAllocation:", allocation)
}

@(objc_type=MTLResidencySet, objc_name="commit")
MTLResidencySet_commit :: #force_inline proc "c" (self: ^MTLResidencySet) {
    objcMsgSend(nil, self, "commit")
}

@(objc_type=MTLResidencySet, objc_name="requestResidency")
MTLResidencySet_requestResidency :: #force_inline proc "c" (self: ^MTLResidencySet) {
    objcMsgSend(nil, self, "requestResidency")
}

@(objc_type=MTLResidencySet, objc_name="endResidency")
MTLResidencySet_endResidency :: #force_inline proc "c" (self: ^MTLResidencySet) {
    objcMsgSend(nil, self, "endResidency")
}

// Missing device helper for creating MTLResidencySet

MTLDevice_makeResidencySet :: #force_inline proc "c" (self: ^mtl.Device, descriptor: ^MTLResidencySetDescriptor) -> (res: ^MTLResidencySet, error: ^ns.Error) {
    res = objcMsgSend(^MTLResidencySet, self, "newResidencySetWithDescriptor:error:", descriptor, &error)
    return
}

MTLCommandQueue_addResidencySet :: #force_inline proc "c" (self: ^mtl.CommandQueue, set: ^MTLResidencySet) {
    objcMsgSend(nil, self, "addResidencySet:", set)
}

// Missing external method for SharedEvent
MTLSharedEvent_wait :: #force_inline proc "c" (self: ^mtl.SharedEvent, untilSignaledValue: u64, timeoutMS: u64) -> mtl.BOOL {
	return objcMsgSend(mtl.BOOL, self, "waitUntilSignaledValue:timeoutMS:", untilSignaledValue, timeoutMS)
}

MTLSamplerDescriptor_lodBias :: #force_inline proc "c" (self: ^mtl.SamplerDescriptor) -> ns.Float
{
    return objcMsgSend(ns.Float, self, "lodBias")
}

MTLSamplerDescriptor_setLodBias :: #force_inline proc "c" (self: ^mtl.SamplerDescriptor, bias: ns.Float)
{
    objcMsgSend(nil, self, "setLodBias:", bias)
}
